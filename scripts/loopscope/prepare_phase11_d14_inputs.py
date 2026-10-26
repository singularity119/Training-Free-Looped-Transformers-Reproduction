#!/usr/bin/env python3
"""Cloud-only D14 inputs: rerender the original gold-free D10 population."""
import argparse
import json
from pathlib import Path

from prepare_phase11_thinking2507_inputs import write_outputs
from tflt.loopscope import phase11_thinking2507 as original
from tflt.loopscope.phase11_d14 import COUNT, engineering_pool, prepare_pool
from tflt.loopscope.phase11_panel import require


def build_inputs(reference, tokenizer):
    original.validate_pool(reference)
    require(not reference['engineering_synthetic'], 'D14 requires original formal D10 inputs')
    pool = prepare_pool(reference, tokenizer)
    require(pool['test_identities'] == reference['test_identities'] and
            pool['rows'] == reference['rows'] and len(pool['rows']) == COUNT,
            'D14 identity/order/options/prompt/input_ids differ from D10')
    return pool, {'status': 'D10_GOLD_FREE_INPUTS_EXACT', 'count': COUNT,
        'identity_order_options_prompt_input_ids_equal': True,
        'target_gold_loaded': False, 'gold_read': False,
        'canonical_source_reused': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--reference-pool', type=Path)
    p.add_argument('--model-provenance-file', type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_dir), local_files_only=True, token=False)
    if a.synthetic:
        pool = prepare_pool(engineering_pool({'scope': 'PREFLIGHT_ONLY',
            'engineering_synthetic': True, 'task': 'gpqa_main', 'target_gold_loaded': False}), tokenizer)
        receipt = {'status': 'ENGINEERING_SYNTHETIC', 'count': 1,
            'source_dataset_read': False, 'gold_read': False}
    else:
        require(a.reference_pool is not None, 'original D10 gold-free pool required')
        pool, receipt = build_inputs(json.loads(a.reference_pool.read_text()), tokenizer)
        receipt['reference_pool'] = str(a.reference_pool)
    provenance = json.loads(a.model_provenance_file.read_text())['model_provenance']
    require(provenance['mode'] == 'USER_DESIGNATED_OSS_UNVERIFIED_REVISION' and
            provenance['model_mount'] == str(a.model_dir), 'D14 designated model differs')
    pool['thinking2507_model_provenance'] = provenance
    receipt.update(model_provenance=provenance, model_dir=str(a.model_dir),
                   max_new_tokens=32768, model_inputs_contain_answer_roles=False)
    write_outputs(a.output_dir, pool, None, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
