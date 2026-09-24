"""Bridge to the engines already built in the sibling `speech2speech` project.

Rationale: PhoWhisper, Gipformer, Parakeet, VieNeu, viXTTS and F5 are already
loaded, measured and debugged over there, several of them out-of-process
because their dependencies conflict. Re-implementing them here would buy
nothing and lose the measurements. The bridge imports that package and dresses
its backends in this platform's protocols, so the conversation plane still sees
only `AsrEngine` / `LlmEngine` / `TtsEngine`.

The dependency points one way: voice-platform -> speech2speech. Nothing in
speech2speech knows this exists.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core.errors import ModelUnavailable

DEFAULT_ROOT = Path(
    os.environ.get("VOICEPLATFORM_S2S_ROOT", "/home/ai01/AIHoang/speech2speech")
)


@lru_cache(maxsize=4)
def load_viet_s2s(root: str | None = None) -> Any:
    """Import `viet_s2s` from the sibling checkout and return the module."""
    base = Path(root) if root else DEFAULT_ROOT
    src = base / "src"
    if not (src / "viet_s2s").is_dir():
        raise ModelUnavailable(
            f"speech2speech checkout not found at {base}. Set "
            "VOICEPLATFORM_S2S_ROOT, or pass options.root, or use a "
            "backend that does not need it."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        # Import both submodules now: a missing optional dependency should
        # surface here as ModelUnavailable, not mid-turn as an ImportError.
        from viet_s2s import backends, config  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env specific
        raise ModelUnavailable(
            f"could not import viet_s2s from {src}: {exc}. Its dependencies "
            "(torch, transformers, onnxruntime) must be importable from the "
            "interpreter running voice-platform."
        ) from exc
    import viet_s2s as module

    # Relative model paths in that project resolve against its own root.
    os.environ.setdefault("VIET_S2S_ROOT", str(base))
    return module


def split_options(options: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    opts = dict(options)
    root = opts.pop("root", None)
    return root, opts
