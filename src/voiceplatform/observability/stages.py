"""Schema-v2 derivation: pair by operation/phrase identity, never across rounds."""
from collections import defaultdict


def measurements(events):
    groups = defaultdict(list)
    phrases = {}
    for event in events:
        if event.data.get('request_id'):
            groups[event.data['request_id']].append(event)
        if event.type.value == 'phrase_ready':
            phrases[event.data['phrase_id']] = {**event.data, 'ready_at_ms': event.ts_ms}
    def first(items, name):
        return min((e.ts_ms for e in items if e.type.value == name), default=None)
    def span(a, b):
        return round(b-a, 3) if a is not None and b is not None and b >= a else None
    rounds, operations = [], []
    for request_id, items in groups.items():
        fields = {}
        for event in items:
            fields.update(event.data)
        start = first(items, 'llm_request_sent')
        end = first(items, 'llm_complete') or first(items, 'llm_terminated')
        queue = next((e.data.get('queue_ms') for e in items if e.type.value == 'model_slot_acquired'), None)
        terminal = next((e.data for e in reversed(items) if e.type.value in ('llm_terminated','model_inference_end')), {})
        row = {'request_id': request_id, 'stage': fields.get('stage'),
               'role': fields.get('role'), 'operation': fields.get('operation'),
               'round': fields.get('round'), 'phrase_id': fields.get('phrase_id'),
               'queue_ms': round(queue,3) if queue is not None else None,
               'outcome': terminal.get('outcome', 'complete' if first(items,'llm_complete') is not None else 'inflight')}
        if row['stage'] == 'llm':
            sent = next((e.data for e in items if e.type.value == 'llm_request_sent'), {})
            row.update({'request_ttft_ms': span(start, first(items,'llm_first_token')),
                        'request_first_tool_ms': span(start, first(items,'llm_first_tool_delta')),
                        'request_total_ms': span(start, end),
                        'prompt_chars': sent.get('prompt_chars'), 'message_count': sent.get('message_count'), 'usage': next((e.data for e in items if e.type.value=='llm_usage'),None)})
            row.update({k:sent.get(k) for k in ('prompt_tokens_counted','prompt_tokens_before_trim','prompt_budget_tokens','prompt_groups_dropped','prompt_count_basis','prompt_prepare_ms') if k in sent})
            rounds.append(row)
        else:
            if not any(e.type.value in ('model_queued','model_inference_start','model_inference_end') for e in items):
                continue  # Finalize wrapper / cached PCM has no native inference.
            row.update({'compute_ms': terminal.get('compute_ms'), 'audio_ms': fields.get('audio_ms'),
                        'rtf': terminal.get('rtf'),
                        'first_chunk_ms': span(first(items,'model_inference_start'),first(items,'tts_chunk_ready')),
                        'lock_wait_ms': next((e.data.get('lock_wait_ms') for e in items if e.type.value=='tts_lock_acquired'),None)})
            operations.append(row)
    for phrase_id, row in phrases.items():
        items = [e for e in events if e.data.get('phrase_id') == phrase_id]
        native = next((r for r in operations if r['phrase_id'] == phrase_id), {})
        row.update({k:native.get(k) for k in ('queue_ms','first_chunk_ms','compute_ms','audio_ms','rtf','lock_wait_ms')})
        row['audio_sent_at_ms'] = first(items,'audio_sent')
        render = [e for e in items if e.data.get('source') == 'browser_audio_render' and e.data.get('clock_uncertainty_ms',1000) <= 20]
        row['playback_started_at_ms'] = first(render,'playback_started')
        row['playback_stopped_at_ms'] = first(render,'playback_stopped')
        row['playback_signal_at_ms'] = first(render,'playback_signal_started')
        complete = next((e.data for e in items if e.type.value=='tts_phrase_complete'),{})
        row['phrase_wait_ms'] = complete.get('phrase_wait_ms')
        row['total_ms'] = complete.get('total_ms')
        row['underruns'] = sum(e.type.value=='playback_underrun' for e in render)
        row['gap_ms'] = sum(e.data.get('gap_ms',0) for e in render if e.type.value=='playback_resumed')
    # A finished phrase followed by an empty queue is a different gap from
    # running out of PCM inside one phrase. Never mix filler -> content here.
    previous = None
    for row in phrases.values():
        row['previous_content_gap_ms'] = None
        if row.get('role') != 'content':
            previous = None
            continue
        if previous is not None:
            row['previous_content_gap_ms'] = span(previous.get('playback_stopped_at_ms'), row.get('playback_started_at_ms'))
        previous = row
    return sorted(rounds,key=lambda r:r.get('round') or 0), operations, list(phrases.values())
