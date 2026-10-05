"""Evaluate the frozen technical release gates without relabeling failures."""
import argparse
import hashlib
import json
from pathlib import Path


def evaluate(suite_path, baseline_path, candidate_path):
    suite = json.loads(suite_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    expected = [c['id'] for c in suite['cases']]
    digest = hashlib.sha256(suite_path.read_bytes()).hexdigest()

    def score(result):
        rows = result['quality']
        groups = {}
        for group in sorted({c['group'] for c in suite['cases']}):
            selected = [r for r in rows if r['group'] == group]
            groups[group] = {'attempted': len(selected),
                             'valid': sum(r['valid'] for r in selected),
                             'accuracy': sum(r['valid'] for r in selected)/len(selected) if selected else 0}
        negatives = [r for r in rows if r['group'] == 'no_tool']
        false_calls = sum(bool(r['calls']) for r in negatives)
        return {'groups': groups, 'attempted': len(rows), 'valid': sum(r['valid'] for r in rows),
                'accuracy': sum(r['valid'] for r in rows)/len(rows) if rows else 0,
                'false_tool_rate': false_calls/len(negatives) if negatives else 1,
                'false_tool_calls': false_calls, 'false_tool_denominator': len(negatives),
                'runtime_errors': [{'id': r['id'], 'error': r['error']} for r in rows if r['error']],
                'failed_cases': [{'id': r['id'], 'question': r['question'], 'answer': r['answer'],
                                  'calls': r['calls'], 'error': r['error']} for r in rows if not r['valid']]}

    b, c = score(baseline), score(candidate)
    gate = suite['gate']
    checks = {
        'same_frozen_cases': all([r['id'] for r in result['quality']] == expected and
                                 result['meta']['suite_sha256'] == digest for result in (baseline, candidate)),
        'same_prompt_tools_options': all(baseline['meta'][k] == candidate['meta'][k]
                                         for k in ('system', 'tools', 'options', 'source_sha256')),
        'runtime_complete': not b['runtime_errors'] and not c['runtime_errors'],
        'direct_accuracy': c['groups']['direct']['accuracy'] >= gate['direct_accuracy_min'],
        'clock_recall': c['groups']['clock']['accuracy'] >= gate['tool_recall_min'],
        'search_recall': c['groups']['search']['accuracy'] >= gate['tool_recall_min'],
        'false_tool_rate': c['false_tool_rate'] <= gate['false_tool_rate_max'],
        'memory_accuracy': c['groups']['memory']['accuracy'] >= gate['memory_accuracy_min'],
        'empirical_noninferiority': b['accuracy'] - c['accuracy'] <= gate['overall_noninferiority_max_drop'],
    }
    if 'language_accuracy_min' in gate:
        checks['language_proxy'] = c['groups']['language']['accuracy'] >= gate['language_accuracy_min']
    return {'suite_sha256': digest, 'gate': gate, 'checks': checks, 'accepted': all(checks.values()),
            'baseline': b, 'candidate': c,
            'scope': 'Empirical gates on the frozen technical cases. No population noninferiority or human conversation/listening claim; CJK absence is only a language regression proxy.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, type=Path)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    decision = evaluate(args.suite, args.baseline, args.candidate)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    raise SystemExit(0 if decision['accepted'] else 1)
