"""Identity captured at process start, never recomputed from a health request."""
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def identity(config):
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    digest = hashlib.sha256()
    for file in sorted((root / "src").rglob("*.py")):
        digest.update(str(file.relative_to(root)).encode())
        digest.update(file.read_bytes())
    versions = {}
    for name in ("fastapi", "uvicorn", "httpx", "numpy", "onnxruntime", "zerotts"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "revision": revision, "dirty": dirty, "source_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "interpreter": sys.executable, "packages": versions,
    }
