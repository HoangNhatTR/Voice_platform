"""Reproduce G2 native-budget/cache evidence alongside the G1 pipeline summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_g1 import stats, summarize


def native_evidence(directory: Path) -> dict:
    cases = [json.loads(p.read_text()) for p in sorted(directory.glob('*-s[123].json'))]
    requests = [r for c in cases for t in c.get('turns', []) for r in t['llm_rounds']]
    completed = [r for r in requests if r['outcome'] == 'complete']
    pairs = [r for r in completed if isinstance(r.get('prompt_tokens_counted'), int)
             and isinstance((r.get('usage') or {}).get('prompt_tokens'), int)]
    native = [(r.get('usage') or {}).get('native_timings', {}) for r in completed]
    content_phrases = [p for c in cases for t in c.get('turns', [])
                       for p in t.get('phrases', []) if p['role'] == 'content']
    return {
        'requests_attempted': len(requests), 'requests_complete': len(completed),
        'prompt_counts_with_usage': len(pairs),
        'prompt_count_matches_usage': sum(r['prompt_tokens_counted'] == r['usage']['prompt_tokens'] for r in pairs),
        'prompt_count_mismatches': [r['request_id'] for r in pairs
                                    if r['prompt_tokens_counted'] != r['usage']['prompt_tokens']],
        'requests_trimmed': sum((r.get('prompt_groups_dropped') or 0) > 0 for r in completed),
        'requests_over_budget': [r['request_id'] for r in completed
                                if (r.get('prompt_tokens_counted') or 0) > (r.get('prompt_budget_tokens') or 10**9)],
        'prompt_prepare_ms': stats([r.get('prompt_prepare_ms') for r in completed]),
        'prompt_tokens_counted': stats([r.get('prompt_tokens_counted') for r in completed]),
        'native_cached_tokens': stats([r.get('cache_n') for r in native]),
        'native_new_prompt_tokens': stats([r.get('prompt_n') for r in native]),
        'native_prefill_ms': stats([r.get('prompt_ms') for r in native]),
        'native_decode_ms': stats([r.get('predicted_ms') for r in native]),
        'content_phrases': len(content_phrases),
        'content_phrases_missing_playback': sum(p.get('playback_started_at_ms') is None for p in content_phrases),
        'content_phrases_negative_playback_delay': sum(
            p.get('playback_started_at_ms') is not None and p.get('audio_sent_at_ms') is not None
            and p['playback_started_at_ms'] < p['audio_sent_at_ms'] for p in content_phrases),
        'scope': 'All attempted requests retained; latency distributions from summarize_g1 exclude unsuccessful content. Native KV cache contains model state, never another session message added to a request.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = {p.name: {**summarize(p), 'native_evidence': native_evidence(p)} for p in args.directories}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: {'attempted': d['cases'], 'valid': d['valid_cases'],
                              'prompt_count_matches': d['native_evidence']['prompt_count_matches_usage']}
                      for name, d in result.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
