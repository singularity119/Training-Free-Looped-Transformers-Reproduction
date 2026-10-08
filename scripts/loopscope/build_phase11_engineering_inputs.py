#!/usr/bin/env python3
"""Build fresh synthetic Phase 11 pools using only a cached tokenizer on CPU."""
import argparse
import json
from pathlib import Path
from tflt.loopscope.phase11_data import MODEL, MODEL_REVISION, SCHEMA, chat_input, render_generation_prompt
from tflt.loopscope.phase11_panel import DATASETS, write_json_once


def build(tokenizer, max_length=262144):
    from lm_eval.api.model import TemplateLM
    from tflt.loopscope.phase10_data import describe_choices
    class Encoder:
        backend = 'causal'
        _encode_pair = TemplateLM._encode_pair
        def tok_encode(self, text, **kwargs):
            return tokenizer.encode(text, add_special_tokens=False)
    pools = {}
    for task in DATASETS:
        rows = []
        for index, (question, choices) in enumerate([
            ('Which option names a shape with three sides?', ['triangle', 'circle']),
            ('Which option names a colour?', ['green', 'square'])]):
            identity = f'phase11-synthetic:{task}:{index}'
            row = {'identity': identity, 'sample_id': identity, 'task': task, 'split': 'synthetic',
                   'source_index': index, 'source_id': index, 'subject': 'synthetic',
                   'question': question, 'choices': choices, 'choice_labels': ['A', 'B'],
                   'fewshot_sample_ids': []}
            if task == 'arc_challenge':
                row['prompt'] = 'Synthetic engineering input only.\nQuestion: '+question+'\nAnswer:'
                meta = describe_choices(row['prompt'], choices, Encoder(), max_length,
                                        tokenizer.all_special_ids, tokenizer.pad_token_id)
                row.update(tokenization={MODEL: meta}, prompt_token_lengths={MODEL: meta['context_length']},
                           prompt_token_length=meta['context_length'],
                           candidate_text_lengths=[len(c) for c in choices])
            else:
                row.update(chat_input(render_generation_prompt({'question': 'Synthetic engineering input only. '+question,
                       'options': choices, 'category': 'synthetic'}), tokenizer, max_length))
            rows.append(row)
        pools[task] = {'schema_version': SCHEMA, 'task': task, 'model': MODEL,
                      'model_revision': MODEL_REVISION, 'tokenizer_revision': MODEL_REVISION,
                      'dataset_revision': None, 'split': 'synthetic', 'scope': 'PREFLIGHT_ONLY',
                      'engineering_synthetic': True, 'target_gold_loaded': False, 'rows': rows,
                      'test_identities': [r['identity'] for r in rows]}
    return pools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('fresh synthetic directory required')
    if args.dry_run:
        print(json.dumps({'scope': 'PREFLIGHT_ONLY', 'tasks': list(DATASETS), 'samples_per_task': 2,
                          'tokenizer_access': False, 'model_access': False, 'gold_access': False}))
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=True)
    pools = build(tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for task, pool in pools.items():
        write_json_once(args.output_dir / (task+'-pool.json'), pool)
    print(json.dumps({'scope': 'PREFLIGHT_ONLY', 'counts': {t: len(p['rows']) for t,p in pools.items()},
                      'output_dir': str(args.output_dir), 'target_gold_loaded': False}))

if __name__ == '__main__':
    main()
