"""Full D6 normal-path runtime/input closure and resource summary; no gold reads."""
import csv
import datetime
import io
import json
from collections import Counter
from pathlib import Path

from phase11_d6 import binding, load_job, validate_bundle
from run_phase11_accuracy import source_provenance
from tflt.loopscope.phase11_data import INSTRUCTION, DIRECT_INSTRUCTION


def close(b):
    stage = Path(b['stage'])
    plan = json.loads((stage / 'plans/execution-plan.json').read_text())
    assert source_provenance(b['source_commit'])['source_head'] == b['source_commit']
    original_path = '/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope/inputs/phase11-gate-a-20261008T094044Z/gpqa_main-prepared/gpqa_main-pool.json'
    original = json.loads(Path(original_path).read_text())
    pools = {mode: json.loads((Path(b['input_root']) / ('formal/' + mode + '-pool.json')).read_text())
             for mode in ('cot', 'direct')}
    assert original['rows'] == pools['cot']['rows']
    assert len(original['rows']) == 448
    identities = [r['identity'] for r in original['rows']]
    assert len(set(identities)) == 448
    for a, c in zip(original['rows'], pools['direct']['rows']):
        assert a['prompt'].count(INSTRUCTION) == 1
        assert c['prompt'] == a['prompt'].replace(INSTRUCTION, DIRECT_INSTRUCTION, 1)
        assert {k:v for k,v in a.items() if k not in ('prompt','input_ids','prompt_token_length')} == {
            k:v for k,v in c.items() if k not in ('prompt','input_ids','prompt_token_length')}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(original['model'], revision=original['tokenizer_revision'],
                                              local_files_only=True, trust_remote_code=True)
    for pool in pools.values():
        assert pool['test_identities'] == identities
        for row in pool['rows']:
            assert tokenizer.encode(row['prompt'], add_special_tokens=False) == row['input_ids']
    checked, counts, per_cell, seeds = [], Counter(), {}, Counter()
    memory, batches = [], []
    pairs = set()
    for task in plan['tasks']:
        batch = json.loads((Path(task['batch_root']) / 'batch_summary.json').read_text())
        assert batch['status'] == 'BATCH_COMPLETE' and batch['packing'] == 5
        assert all(f['exit_code'] == 0 for f in batch['facts'])
        batches.append({'task':task['number'],'job_id':batch['job_id'],'elapsed_seconds':batch['elapsed_seconds'],
                        'packing':5,'process_count':len(batch['facts'])})
        for path in Path(task['batch_root']).glob('attempts/*/device_memory.jsonl'):
            for line in path.read_text().splitlines():
                z = json.loads(line)
                assert z['exit_code'] == 0
                row = next(csv.reader(io.StringIO(z['devices_csv'])))
                memory.append((float(row[2]), float(row[3])))
        for job in task['jobs']:
            scope, config, cell, pool, indices = load_job(job)
            metadata = binding(job, scope, config, cell, indices)
            root = Path(job['run_root'])
            assert json.loads((root/'command_args.json').read_text()) == metadata
            summary = json.loads((root/'summary.json').read_text())
            assert summary['status'] == 'RAW_COMPLETE' and summary['count'] == len(indices)
            assert metadata['scope'] == 'FORMAL_TEST' and metadata['max_new_tokens'] == 8192
            if cell['arm'] != 'Native':
                assert cell['decode_mode'] == 'full' and cell['cache_strategy'] == 'first'
            envs = list((root/'attempts').glob('*/env.json'))
            assert len(envs) == 1, 'unexpected attempt count requires separate diagnosis'
            env = json.loads(envs[0].read_text())
            assert env['source_head'] == b['source_commit']
            assert env['effective_rng_seeds'] == {k:20261015 for k in ('python','numpy','torch','cuda_all_devices')}
            seeds[20261015] += 1
            bundles = [json.loads(path.read_text()) for path in sorted((root/'complete').glob('*.json'))]
            assert len(bundles) == len(indices)
            raw = [json.loads(l) for l in (root/'records.jsonl').read_text().splitlines() if l.strip()]
            tel = [json.loads(l) for l in (root/'telemetry.jsonl').read_text().splitlines() if l.strip()]
            assert [z['record'] for z in bundles] == raw and [z['telemetry'] for z in bundles] == tel
            for i,z in zip(indices,bundles):
                validate_bundle(i,z['record'],z['telemetry'],metadata,pool)
                pair = (job['cell_key'],i)
                assert pair not in pairs
                pairs.add(pair)
                assert z['record']['identity'] == identities[i]
                counts[job['cell_key']] += 1
                t = z['telemetry']
                entry = per_cell.setdefault(job['cell_key'], {'peak_allocated_bytes':0,'peak_reserved_bytes':0,'sum_question_elapsed_seconds':0})
                for key in ('peak_allocated_bytes','peak_reserved_bytes'):
                    entry[key] = max(entry[key],t[key])
                entry['sum_question_elapsed_seconds'] += t['elapsed_seconds']
            checked.append({'run_root':str(root),'cell_key':job['cell_key'],'count':len(indices),'source_commit':b['source_commit']})
    assert len(pairs) == 5376 and len(counts) == 12 and set(counts.values()) == {448}
    assert len(checked) == 108 and sum(seeds.values()) == 108
    peak = max(memory)
    result = {'schema':'loopscope.phase11.d6.runtime_closure.v1','status':'ALL_RAW_RUNTIME_INPUTS_CLOSED_GOLD_UNREAD',
        'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':b['source_commit'],
        'record_count':5376,'cell_counts':dict(counts),'process_roots_checked':108,'effective_seed_process_counts':dict(seeds),
        'cot_original_rows_identical':True,'direct_only_exact_instruction_replaced':True,'prompt_tokenizations_checked':896,
        'shared_identity_count':448,'scope_lifecycle_validated':True,'ordinary_loop_decode':'full','cache_strategy':'first',
        'max_new_tokens':8192,'runtime_seed':20261015,'input_seed':20261008,'per_cell_resources':per_cell,
        'device_peak_used_MiB':peak[0],'device_memory_total_MiB':peak[1],
        'free_fraction_at_device_peak':1-peak[0]/peak[1],'device_memory_samples':len(memory),'batches':batches,
        'run_roots':checked,'target_gold_loaded':False}
    out = Path(b['run_root'].replace('/runs/','/artifacts/'))
    out.mkdir(parents=True,exist_ok=False)
    with (out/'runtime_closure.json').open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result))
