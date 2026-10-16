#!/usr/bin/env python3
"""CPU-only D7 resource projections. Never extract answers or read sealed gold.

Plans identify the exact jobs. Atomic bundles are projected immediately onto
length/end-reason/timing/memory fields; generated text and answers are unused.
The four prompt-quantile canary questions are not a representative runtime
sample. Cost ranges below are scenarios, not confidence intervals.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path


CELLS = ('native', 'current_t_k2_l05', 'fixed_t0_k3_l09', 'loop_k2', 'loop_k3')
PROXIES = {'loop_k2': 'current_t_k2_l05', 'loop_k3': 'fixed_t0_k3_l09'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def distribution(values):
    if not values:
        return None
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('nonfinite or negative resource observation')
    return {'n': len(values), 'mean': sum(values) / len(values), 'min': min(values),
            'median': quantile(values, .5), 'p90': quantile(values, .9), 'max': max(values)}


def project_bundle(path):
    bundle = read(path)
    record, telemetry = bundle['record'], bundle['telemetry']
    return {**{k: telemetry[k] for k in ('canonical_index', 'generated_token_count',
                'prefill_input_tokens', 'elapsed_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')},
            **{k: telemetry[k] for k in ('prefill_elapsed_seconds', 'decode_elapsed_seconds') if k in telemetry},
            'end_reason': record['end_reason']}


def device_samples(path):
    by_device = {}
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['exit_code']:
                raise ValueError('failed device memory telemetry: ' + str(path))
            for values in csv.reader(row['devices_csv'].splitlines(), skipinitialspace=True):
                if len(values) != 4:
                    raise ValueError('unexpected nvidia-smi memory row')
                uuid, name, used, total = [v.strip() for v in values]
                used, total = float(used) * 2**20, float(total) * 2**20
                fact = by_device.setdefault(uuid, {'uuid': uuid, 'name': name,
                    'total_bytes': total, 'peak_used_bytes': 0, 'samples': 0})
                if fact['total_bytes'] != total:
                    raise ValueError('device memory capacity changed')
                fact['peak_used_bytes'] = max(fact['peak_used_bytes'], used)
                fact['samples'] += 1
    return list(by_device.values())


def kv_bound(config, max_prompt, max_new=8192):
    config = config.get('text_config', config)
    layers = int(config['num_hidden_layers'])
    kv_heads = int(config.get('num_key_value_heads', config['num_attention_heads']))
    head_dim = int(config.get('head_dim', config['hidden_size'] // config['num_attention_heads']))
    # Native/first-cache loop keeps one K/V pair per physical layer. BF16=2 B.
    per_token = 2 * layers * kv_heads * head_dim * 2
    return {'num_hidden_layers': layers, 'num_key_value_heads': kv_heads,
            'head_dim': head_dim, 'dtype_bytes': 2, 'batch_size': 1,
            'max_prompt_tokens': max_prompt, 'max_new_tokens': max_new,
            'kv_bytes_per_token': per_token,
            'full_context_kv_bytes': per_token * (max_prompt + max_new),
            'loop_cache_multiplier': 1}


def summarize(plan_paths, model_config, max_prompt_tokens, scheduler_path=None,
              population=12032, packing_cap=8):
    cells = defaultdict(list)
    envs, devices, batch_facts, memberships = [], {}, [], defaultdict(set)
    roots_seen = set()
    single_process_device_peaks = []
    for plan_path in plan_paths:
        plan = read(plan_path)
        for task in plan['tasks']:
            batch_root = Path(task['batch_root'])
            for telemetry_path in batch_root.glob('attempts/*/device_memory.jsonl'):
                for device in device_samples(telemetry_path):
                    if task['packing'] == 1:
                        single_process_device_peaks.append(device['peak_used_bytes'])
                    identity = device['uuid']
                    if identity not in devices:
                        devices[identity] = device
                    else:
                        devices[identity]['peak_used_bytes'] = max(devices[identity]['peak_used_bytes'], device['peak_used_bytes'])
                        devices[identity]['samples'] += device['samples']
            for path in batch_root.glob('attempts/*/batch_summary.json'):
                value = read(path)
                batch_facts.append({k: value[k] for k in ('job_id', 'status', 'packing', 'elapsed_seconds')})
            for job in task['jobs']:
                if job['synthetic'] or job['engineering']:
                    raise ValueError('formal MMLU-Pro resource report cannot include synthetic/engineering jobs')
                key, root = job['cell_key'], Path(job['run_root'])
                if key not in CELLS:
                    raise ValueError('unknown D7 cell')
                if str(root) in roots_seen:
                    continue
                roots_seen.add(str(root))
                for path in root.glob('attempts/*/env.json'):
                    value = read(path)
                    envs.append({k: value[k] for k in ('gpu', 'total_memory', 'job_id')})
                for path in sorted((root / 'complete').glob('*.json')):
                    row = project_bundle(path)
                    index = row['canonical_index']
                    if index not in job['indices'] or index in memberships[key]:
                        raise ValueError('record outside plan membership or duplicate cell/index')
                    memberships[key].add(index)
                    cells[key].append(row)
    if not envs or not cells:
        raise ValueError('no completed formal resource observations and environment')
    bound = kv_bound(read(model_config), max_prompt_tokens)
    per_cell = {}
    for key, rows in sorted(cells.items()):
        stats = {name: distribution([r[name] for r in rows if name in r]) for name in
                 ('generated_token_count', 'prefill_input_tokens', 'elapsed_seconds',
                  'prefill_elapsed_seconds', 'decode_elapsed_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
        if stats['prefill_input_tokens']['max'] > max_prompt_tokens:
            raise ValueError('observed prompt exceeds declared frozen maximum')
        per_cell[key] = {'record_count': len(rows), 'canonical_indices': sorted(memberships[key]),
            'statistics': stats, 'end_reason_counts': dict(Counter(r['end_reason'] for r in rows)),
            'padded_process_memory_bytes': stats['peak_reserved_bytes']['max'] + bound['full_context_kv_bytes']}
    device_total = min([e['total_memory'] for e in envs] + [d['total_bytes'] for d in devices.values()])
    worst = max(v['padded_process_memory_bytes'] for v in per_cell.values())
    if single_process_device_peaks:
        # A packing-1 device trace includes CUDA context/nonallocator overhead.
        worst = max(worst, max(single_process_device_peaks) + bound['full_context_kv_bytes'])
    candidate = min(packing_cap, math.floor(.8 * device_total / worst))
    if candidate < 1:
        raise ValueError('one process plus full KV padding exceeds 80 percent memory budget')
    scenarios = {}
    for key in CELLS:
        source = key if key in cells else PROXIES.get(key)
        if source not in cells:
            scenarios[key] = {'status': 'UNOBSERVED_NO_PROXY'}
            continue
        rows = cells[source]
        durations = [r['elapsed_seconds'] for r in rows]
        # Scale decode component to the full 8192-token ceiling where timing is
        # available; otherwise scale total time, conservatively including prefill.
        ceiling = []
        for row in rows:
            count = max(1, row['generated_token_count'] - 1)
            if 'decode_elapsed_seconds' in row:
                decode = row['decode_elapsed_seconds']
                ceiling.append(max(row['elapsed_seconds'], row['elapsed_seconds'] - decode + decode * 8191 / count))
            else:
                ceiling.append(row['elapsed_seconds'] * 8192 / max(1, row['generated_token_count']))
        scenarios[key] = {'observed_cell': source, 'proxy_not_measured': source != key,
            'sample_mean_serial_process_hours': population * sum(durations) / len(durations) / 3600,
            'observed_min_max_serial_process_hours': [population * min(durations) / 3600,
                                                      population * max(durations) / 3600],
            'all_questions_8192_serial_process_hours': population * max(ceiling) / 3600}
    estimated = [v for v in scenarios.values() if 'sample_mean_serial_process_hours' in v]
    full = len(estimated) == 5
    nominal = sum(v['sample_mean_serial_process_hours'] for v in estimated)
    ceiling = sum(v['all_questions_8192_serial_process_hours'] for v in estimated)
    scheduler = read(scheduler_path) if scheduler_path else None
    return {'schema': 'loopscope.phase11.d7.resources.v1', 'target_gold_loaded': False,
        'generated_text_or_answers_extracted': False, 'plans': list(map(str, plan_paths)),
        'model_config_path': str(model_config), 'population_per_cell': population,
        'target_record_count': population * 5, 'cells': per_cell, 'environments': envs,
        'devices': list(devices.values()), 'batch_attempts': batch_facts, 'kv_bound': bound,
        'packing': {'candidate_from_memory_bound': candidate, 'device_capacity_bytes': device_total,
            'headroom_fraction': .2, 'largest_padded_process_bytes': worst,
            'single_process_device_peak_bytes': max(single_process_device_peaks) if single_process_device_peaks else None,
            'method': 'max(observed peak reserved, packing-1 device peak when available) plus entire worst-context KV; observed KV is intentionally counted twice',
            'validated_concurrency': sorted(set(v['packing'] for v in batch_facts if v['status'] == 'BATCH_COMPLETE')),
            'limitation': 'Candidate is a conservative KV-based estimate, not validated packed execution. Prefill/SVD/attention workspace and allocator competition require bounded first packed execution.'},
        'cost_scenarios_by_cell': scenarios,
        'cost_projection': {'full_five_cell_projection': full, 'assumed_packing': candidate,
            'sample_mean_allocation_gpuh': nominal / candidate if full else None,
            'all_8192_scenario_allocation_gpuh': ceiling / candidate if full else None,
            'ideal_8_gpu_wall_hours_scenarios': [nominal / candidate / 8, ceiling / candidate / 8] if full else None,
            'limitation': 'Conditional scenarios, not statistical intervals; prompt-quantile n=4 is not a random runtime sample, output lengths may differ by method, Loop proxies are unmeasured, and queues/startup/tails/stage barriers can increase wall time.'},
        'selection_bias': {'design': 'four prompt-length quantiles .25/.5/.75/1, tie-break canonical index',
            'caveat': 'Prompt length is not generated length; maximum-prompt early EOS does not validate worst 8192 memory or typical throughput. n=4 output lengths cannot establish population runtime quantiles.'},
        'scheduler_records': scheduler}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='append', required=True)
    parser.add_argument('--model-config', required=True)
    parser.add_argument('--max-prompt-tokens', required=True, type=int)
    parser.add_argument('--scheduler', help='gold-free observe_phase11_d7 JSON')
    parser.add_argument('--population', type=int, default=12032)
    parser.add_argument('--packing-cap', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(summarize(args.plan, args.model_config, args.max_prompt_tokens,
        args.scheduler, args.population, args.packing_cap), allow_nan=False))
