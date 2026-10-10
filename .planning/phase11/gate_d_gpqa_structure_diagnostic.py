"""Bounded gold-free structural diagnosis; never emits text or candidate letters."""
import collections
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path('/hpc2hdd/home/xhuang225/workspaces/training_free_looped_transformers_loopscope')
STAMP = 'phase11-gate-d-20261009T023200Z'
SOURCE = ROOT / 'staging' / STAMP / 'source'
REV = 'cdbee75f17c01a7cc42f958dc650907174af0554'
APPROVED = {'447f64be3172dae403f7cd0c07ae17399a6197b4', 'f9845ea86e3deaae4145fd383d1f316a70d7dfe3'}
OUT = ROOT / 'runs' / (STAMP + '-gpqa-format-structure-supplement.json')
PHRASE = re.compile(r'\bfinal\s+answer\b', re.I)
PAREN = re.compile(r'final\s+answer\s*[:\uff1a]\s*\(\s*[A-Da-d]\s*\)\s*[.!;,\uff0e\uff01\u3002]?\s*', re.I)
BARE = re.compile(r'final\s+answer\s*[:\uff1a]\s*[A-Da-d]\s*[.!;,\uff0e\uff01\u3002]?\s*', re.I)
ANSWER_CUE = re.compile(r'\b(?:answer|correct\s+option|correct\s+choice)\b|\\boxed', re.I)
ORDER = ['markdown_wrapped_answer_line', 'parenthesized_case_spacing_punctuation',
         'bare_letter_answer_line', 'final_answer_phrase_prose_or_incomplete',
         'other_answer_cue_without_final_phrase', 'missing_all_answer_cues']

def classify(text):
    lines = text.splitlines()
    flags = {'final_answer_phrase': bool(PHRASE.search(text)),
             'other_answer_cue': bool(ANSWER_CUE.search(text)),
             'boxed_marker': '\\boxed' in text,
             'markdown_wrapped_answer_line': False,
             'parenthesized_flexible_line': False,
             'bare_letter_flexible_line': False,
             'case_difference_on_phrase_line': False,
             'markdown_marker_on_phrase_line': False}
    for line in lines:
        s = line.strip()
        if PHRASE.search(s):
            flags['case_difference_on_phrase_line'] |= 'Final answer' not in s
            flags['markdown_marker_on_phrase_line'] |= bool(re.search(r'[*_`~]|^\s*(?:#{1,6}|>)', s))
        flags['parenthesized_flexible_line'] |= bool(PAREN.fullmatch(s))
        flags['bare_letter_flexible_line'] |= bool(BARE.fullmatch(s))
        plain = re.sub(r'[*_`~]', '', s)
        plain = re.sub(r'^\s*(?:[-+]|\d+[.)]|#{1,6}|>)\s*', '', plain).strip()
        flags['markdown_wrapped_answer_line'] |= (plain != s and bool(PAREN.fullmatch(plain) or BARE.fullmatch(plain)))
    if flags['markdown_wrapped_answer_line']:
        category = ORDER[0]
    elif flags['parenthesized_flexible_line']:
        category = ORDER[1]
    elif flags['bare_letter_flexible_line']:
        category = ORDER[2]
    elif flags['final_answer_phrase']:
        category = ORDER[3]
    elif flags['other_answer_cue']:
        category = ORDER[4]
    else:
        category = ORDER[5]
    return category, flags

# Synthetic structure-only assertions. No alternate answer parser or score.
assert classify('**Final Answer: (X)**'.replace('X', 'A'))[0] == ORDER[0]
assert classify('Final Answer : ( A ).')[0] == ORDER[1]
assert classify('Final answer: A.')[0] == ORDER[2]
assert classify('The final answer is discussed here.')[0] == ORDER[3]
assert classify('The answer is discussed here.')[0] == ORDER[4]
assert classify('unfinished reasoning')[0] == ORDER[5]

closure = json.load(open(ROOT / 'runs' / (STAMP + '-gpqa-full-raw-closure.json')))
assert closure['status'] == 'PANEL_CLOSED_GOLD_UNREAD'
assert closure['independent_cell_count'] == 18 and closure['sample_count'] == 448
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'], text=True).strip() == 'f9845ea86e3deaae4145fd383d1f316a70d7dfe3'
assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'], text=True).strip()
sys.path.insert(0, str(SOURCE / 'src'))
from tflt.loopscope.phase11_data import INSTRUCTION, chat_input, render_generation_prompt
os.environ['HF_HOME'] = '/hpc2hdd/home/xhuang225/shared/hf_home'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer
pool = json.load(open(ROOT / 'inputs/phase11-gate-a-20261008T094044Z/gpqa_main-prepared/gpqa_main-pool.json'))
assert pool['model_revision'] == pool['tokenizer_revision'] == REV
rows = pool['rows']
assert len(rows) == 448
# Tokenizer only, local cache only. No model loading, CUDA or network download.
tok = AutoTokenizer.from_pretrained(pool['model'], revision=REV, local_files_only=True, trust_remote_code=True)
prompt_checks = collections.Counter()
for row in rows:
    target = {'question':row['question'], 'options':row['choices'], 'category':'gpqa_main'}
    reconstructed = chat_input(render_generation_prompt(target), tok, 40960)
    assert reconstructed['prompt'] == row['prompt']
    assert reconstructed['input_ids'] == row['input_ids']
    assert tok.encode(row['prompt'], add_special_tokens=False) == row['input_ids']
    assert INSTRUCTION in row['prompt'] and row['fewshot_sample_ids'] == []
    assert row['prompt'].count('<|im_start|>user\n') == 1
    assert '<|im_start|>system\n' not in row['prompt']
    assert row['prompt'].endswith('<|im_start|>assistant\n')
    prompt_checks.update(['native_template_exact_text', 'native_template_exact_ids', 'encode_exact_ids',
                          'exact_instruction', 'zero_shot', 'one_user_no_system', 'assistant_generation_prefix'])

roots = json.load(open(ROOT / 'inputs' / STAMP / 'gpqa-full-cell-roots.json'))
assert len(roots) == 18 and sum(map(len,roots.values())) == 579
result = {group: {'total':0,'strict_valid':0,'strict_failed':0,'failure_categories':collections.Counter(),
                  'overlapping_failure_flags':collections.Counter()} for group in ('native_eos','length_limit')}
seen = set(); metadata = collections.Counter(); strict = collections.Counter(); source_roots = collections.Counter()
for cell, paths in roots.items():
    for path in paths:
        root = pathlib.Path(path)
        env = json.load(open(root / 'env.json'))
        args = json.load(open(root / 'command_args.json'))
        assert env['source_commit'] in APPROVED and args['source_commit'] == env['source_commit']
        assert env['model_revision'] == env['tokenizer_revision'] == REV
        assert env['native_generation_eos_ids'] == [151645,151643]
        assert env['dtype'] == 'torch.bfloat16'
        assert args['max_new_tokens'] == 2048 and args['engineering_generate_steps'] is None
        assert args['scope'] == 'FORMAL_TEST'
        assert args['cell']['batch_size'] == 1 and args['cell']['decode_mode'] == 'full'
        assert pathlib.Path(args['pool']) == ROOT / 'inputs/phase11-gate-a-20261008T094044Z/gpqa_main-prepared/gpqa_main-pool.json'
        source_roots[env['source_commit']] += 1
        metadata.update(['formal_2048_no_forced_minimum', 'revision_dtype_native_eos', 'frozen_pool', 'batch1_full_decode'])
        with open(root / 'records.jsonl') as handle:
            for line in handle:
                rec = json.loads(line); index = rec['canonical_index']
                assert (cell,index) not in seen and 0 <= index < 448
                seen.add((cell,index))
                assert rec['identity'] == rows[index]['identity'] and rec['status'] == 'OK'
                ids = rec['generated_token_ids']
                assert tok.decode(ids, skip_special_tokens=True) == rec['generated_text']
                assert rec['generated_token_count'] == len(ids) <= 2048
                assert rec['eos_reached'] == (ids[-1] in (151645,151643))
                assert rec['truncated'] == (rec['end_reason'] == 'length_limit')
                assert rec['eos_reached'] or len(ids) == 2048
                group = 'native_eos' if rec['end_reason'] == 'eos' else 'length_limit'
                data = result[group]; data['total'] += 1
                strict[rec['extraction_status']] += 1
                if rec['extraction_status'] == 'unique_valid':
                    data['strict_valid'] += 1
                else:
                    data['strict_failed'] += 1
                    category, flags = classify(rec['generated_text'])
                    data['failure_categories'][category] += 1
                    data['overlapping_failure_flags'].update(k for k,v in flags.items() if v)
                    if not flags['final_answer_phrase']:
                        data['overlapping_failure_flags']['missing_final_answer_phrase'] += 1
                metadata['generated_text_equals_native_decode'] += 1
for cell in roots:
    assert all((cell,index) in seen for index in range(448))
assert len(seen) == 8064 and strict == {'no_valid_final_line':8051,'unique_valid':13}
for data in result.values():
    assert sum(data['failure_categories'].values()) == data['strict_failed']
    assert data['strict_failed'] + data['strict_valid'] == data['total']
    data['failure_categories'] = {name:data['failure_categories'][name] for name in ORDER}
assert result['native_eos']['total'] == 4122 and result['length_limit']['total'] == 3942
report = {'schema':'loopscope.phase11.gpqa_structure_supplement.v1', 'gate':'D',
    'scope':'single_existing_output_cpu_diagnostic_no_gold_no_predictions_no_rescoring',
    'executor':'01a11e77-9550-7092-877d-1ecb30c531de/local',
    'source_compute':'f9845ea86e3deaae4145fd383d1f316a70d7dfe3',
    'source_roots':source_roots, 'record_count':len(seen), 'attempt_root_count':579,
    'strict_status_counts':strict, 'groups':result,
    'category_semantics':'Failures only; mutually exclusive first-match precedence in category_order. Structural booleans never retain candidate letters or alternative predictions. Markdown category may overlap case/bare differences.',
    'category_order':ORDER, 'overlap_semantics':'Flags count failed records once per flag; flags overlap and must not be summed.',
    'structural_patterns':{'phrase':PHRASE.pattern,'parenthesized_line':PAREN.pattern,'bare_line':BARE.pattern,
        'markdown_removal':'Remove *_`~; remove one leading list/heading/blockquote marker; boolean full-line shape test only.',
        'other_answer_cue':ANSWER_CUE.pattern},
    'prompt_checks':prompt_checks, 'metadata_checks':metadata,
    'tokenizer':{'model':pool['model'],'revision':REV,'cache_only':True,'model_loaded':False},
    'code_evidence':['src/tflt/loopscope/phase11_data.py:74-94,137-160',
                     'scripts/loopscope/run_phase11_accuracy.py:117-154,238-280',
                     'src/tflt/loopscope/phase11_accuracy.py:19-49,57-63'],
    'limitations':['Structure classification is descriptive and does not determine reasoning correctness.',
        'Prefix/Markdown/letter shapes are not repaired or rescored.',
        'Stored metadata and immutable producer code establish generation options; no new model run.'],
    'raw_content_exported':False,'gold_read':False,'alternative_predictions_generated':False,
    'gpu_jobs_submitted':False,'frozen_extractor_changed':False}
with OUT.open('x') as handle:
    json.dump(report,handle,indent=2,sort_keys=True);handle.write('\n')
print(json.dumps(report,sort_keys=True))
