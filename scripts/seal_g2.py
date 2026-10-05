"""Seal completed G2 evidence after timed inference and documentation are finished."""
import argparse
import hashlib
import json
import platform
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('docs/audits/2026-09-29/g2'))
    args = parser.parse_args()
    target = args.directory
    assert all(json.loads((target / run / 'manifest.json').read_text())['finished_at']
               for run in ('baseline-1', 'baseline-3'))
    old = json.loads(Path('docs/audits/2026-09-28/g1/model-manifest.json').read_text())
    assets = {Path(row['path']) for row in old['files']}
    assets.update(Path('../speech2speech/models').glob('qwen3-4b/*.gguf'))
    assets.update(Path('../speech2speech/models').glob('qwen3.5-4b/*.gguf'))
    assets.update(Path('models/llm/qwen2.5-7b').glob('*.gguf'))
    records = [{'path': str(p.absolute()), 'resolved_path': str(p.resolve()),
                'bytes': p.stat().st_size, 'sha256': digest(p)} for p in sorted(assets)]
    unchanged = {row['path']: row['sha256'] for row in old['files']}
    changed = [r['path'] for r in records if r['path'] in unchanged and r['sha256'] != unchanged[r['path']]]
    (target / 'model-manifest.json').write_text(json.dumps({
        'hashed_at': datetime.now(timezone.utc).isoformat(), 'hash_timing': 'After timed runs; no model upgrade during benchmarks',
        'host': platform.node(), 'machine': platform.machine(), 'files': records,
        'changed_from_g1': changed, 'zerotts_revision': 'c2bfbd67dc648cac455077333f7cf5c18a2e3bb4',
        'qwen25_download': 'model-download.json', 'origin_limit': 'Existing4B GGUF identified by bytes/hash; upstream conversion revision is not inferred.'
    }, indent=2) + '\n')
    code = [p for root in ('src', 'web', 'tests', 'scripts', 'configs')
            for p in Path(root).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    code += [p for p in Path('.').glob('requirements*') if p.is_file()]
    code += [Path(x) for x in ('README.md', '.gitignore', 'pyproject.toml',
                              'docs/OPERATIONS.md', 'docs/DEVELOPMENT_PLAN_REALTIME.md')]
    code += [target / 'REPORT.md', target / 'IMPLEMENTATION.md']
    code = sorted(set(code))
    with tarfile.open(target / 'source-snapshot.tar.gz', 'w:gz') as archive:
        for path in code:
            archive.add(path, arcname=str(path), recursive=False)
    source = hashlib.sha256()
    for path in sorted(Path('src').rglob('*.py')):
        source.update(str(path).encode()); source.update(path.read_bytes())
    runs = {}
    for path in target.iterdir():
        if not path.is_dir() or path.name in ('logs', 'listening'):
            continue
        if path.name in ('baseline-1', 'baseline-3'):
            scope = 'final fixed-input direct pipeline confirmation'
        elif path.name.startswith('supplementary-') or path.name.startswith('clean-search-'):
            scope = 'supplemental final configuration; failures retained, separate from direct'
        elif path.name in ('release3-4b', 'release3-9b'):
            scope = 'prospective procedural technical quality confirmation; related families, same author'
        elif path.name in ('latency-final-4b', 'latency-final-9b'):
            scope = 'frozen final-prompt native TTFT A/B; 20 one and60 three correlated fixed-history tasks, sequential model order'
        else:
            scope = 'development/exploratory/retired configuration; not final release evidence'
        runs[path.name] = scope
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(),
                'backend_source_sha256': source.hexdigest(),
                'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'workspace_dirty': True, 'code_files': {str(p): digest(p) for p in code},
                'run_scope': runs, 'sealed': 'CHECKSUMS.sha256',
                'scope': 'Source snapshot and hash-addressed local dependencies; no weights/private TLS keys/venv bundled. Package lock is an installed-environment snapshot, not a portable hashed wheel lock.'}
    (target / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files = sorted(p for p in target.rglob('*') if p.is_file() and p.name != 'CHECKSUMS.sha256')
    (target / 'CHECKSUMS.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(target)}\n' for p in files))
    print(json.dumps({'sealed_files': len(files), 'backend_source_sha256': source.hexdigest(),
                      'changed_assets_from_g1': changed}))


if __name__ == '__main__':
    main()
