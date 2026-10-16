#!/usr/bin/env python3
"""D7 synthetic three-token CUDA scope verification, without target data/gold."""
import argparse
import json
import os
from pathlib import Path

from tflt.loopscope.phase11_accuracy import load_pool
from tflt.loopscope.phase11_panel import load_config, require, write_json_once
from tflt.loopscope.phase11_runtime import make_runtime
from tflt.loopscope.phase11_adapter import generation_forward_context
from run_phase11_accuracy import load_runtime, runtime_telemetry, source_provenance
from check_phase11_gpu import WrapperAudit, capture_logits, loop_config


def run(plan_path):
    require(os.environ.get('SLURM_JOB_PARTITION') == 'debug', 'scope check requires debug allocation')
    plan = json.loads(Path(plan_path).read_text())
    number = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    task = next(t for t in plan['tasks'] if t['number'] == number)
    jobs = task['jobs']
    require(all(j['synthetic'] for j in jobs), 'scope check may only use synthetic inputs')
    source = source_provenance(jobs[0]['source_commit'])
    config = load_config()
    from phase11_d7 import selected_cells
    cells = selected_cells()
    base = cells['current_t_k2_l05']
    # The helper may return a flat cell or a cell specification.
    base = base.get('cell', base)
    torch, model, tokenizer, _, _, env = load_runtime(base, config, 'mmlu_pro')
    torch.set_grad_enabled(False)
    pool = load_pool(Path(next(j['pool'] for j in jobs if j['prompt_mode'] == 'cot')),
                     'mmlu_pro', 'PREFLIGHT_ONLY')
    row = pool['rows'][0]
    tokens = row['input_ids']
    require(tokenizer.encode(row['prompt'], add_special_tokens=False) == tokens, 'synthetic tokenizer differs')
    inputs = torch.tensor([tokens], dtype=torch.long, device=model.device)
    mask = torch.ones_like(inputs)
    from tflt.wrapper import looped_model
    evidence = {}
    for key in ('current_t_k2_l05', 'fixed_t0_k3_l09'):
        spec = cells[key]
        cell = spec.get('cell', spec)
        for scope in ('full',):
            runtime = make_runtime({**cell, 'damping_scope': scope})
            collector = WrapperAudit(cell['k'])
            with looped_model(model, loop_config(cell['k'], runtime, collector)), capture_logits(torch, model) as captured:
                with generation_forward_context(model, runtime, row['identity'], tokens,
                    attention_mask=[1] * len(tokens), special_ids=tokenizer.all_special_ids,
                    pad_token_id=tokenizer.pad_token_id) as lifecycle:
                    output = model.generate(input_ids=inputs, attention_mask=mask, do_sample=False,
                        num_beams=1, use_cache=True, min_new_tokens=3, max_new_tokens=3)
            require(output.shape[-1] == len(tokens) + 3 and lifecycle['prefill_count'] == 1
                    and lifecycle['decode_count'] == 2, 'synthetic forward counts differ')
            for i, forward in enumerate(captured['forwards']):
                require(forward['cache_lengths_by_layer'] == [len(tokens) + i] * model.config.num_hidden_layers,
                        'first-KV cache growth differs')
            actual = runtime_telemetry(runtime)
            require(actual['decode_callback_count'] == 2 * cell['k'] and actual['decode_fit_count'] == 0,
                    'decode must execute K callbacks without refitting')
            expected = 0 if scope == 'prefill_only' else 2 * sum(used is not None for _, used, _ in runtime.schedule)
            require(sum(actual['decode_applied_by_t'].values()) == expected, 'actual decode damping scope differs')
            evidence[key + ':' + scope] = {'wrapper': collector.validate(3, lifecycle), 'online': actual}
    result = {'status': 'D7_FULL_DAMPING_CUDA_VERIFIED', 'target_gold_loaded': False,
              'source': source, 'environment': env, 'synthetic_generated_tokens': 3,
              'evidence': evidence,
              'resource_sizing': False}
    write_json_once(Path(task['batch_root']) / 'scope_check.json', result)
    print(json.dumps({'status': result['status'], 'checks': len(evidence), 'target_gold_loaded': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    run(p.parse_args().plan)
