"""Turning model text into speakable text, incrementally.

The non-obvious part is that this has to work on a *stream*. A filter that is
only ever tested on whole messages passes, and then in production the segmenter
hands "[Hồ Gươm](https://" to the talker because the closing bracket had not
arrived yet, and the user hears a URL where the label should have been. So the
filter holds back any tail that could still turn into markup, and releases it
once the construct closes or the stream ends.
"""

from __future__ import annotations

import re

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_CODE_FENCE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BULLET = re.compile(r"^\s{0,4}[-*+]\s+", re.MULTILINE)
_ORDERED = re.compile(r"^\s{0,4}(\d+)[.)]\s+", re.MULTILINE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]",
    flags=re.UNICODE,
)
_MULTISPACE = re.compile(r"[ \t]{2,}")

# Characters that can only be markup mid-construct. After every *complete*
# construct is masked out, the earliest one still standing is where the safe
# prefix has to stop.
_MARKERS = "[`*_"


def strip_markdown(text: str) -> str:
    """Whole-text sanitisation. Use the streaming filter on live output."""
    text = _CODE_FENCE.sub(" ", text)
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _BOLD_ITALIC.sub(r"\2", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _ORDERED.sub(r"\1. ", text)
    text = _EMOJI.sub("", text)
    return _MULTISPACE.sub(" ", text)


def strip_emotion_cues(text: str) -> str:
    """Remove "[cười]"-style cues for a talker that cannot act on them.

    Whether to call this is a property of the loaded engine
    (`TtsCapabilities.emotion_cues`), never a config flag: a config that
    promises cue support the checkpoint lacks makes the cue be read aloud.
    """
    return _MULTISPACE.sub(" ", re.sub(r"\[[^\[\]]{1,24}\]", " ", text)).strip()


class SpeechTextFilter:
    """Incremental markdown stripper with hold-back."""

    def __init__(self, max_hold_chars: int = 160) -> None:
        self._buf = ""
        self.max_hold_chars = max_hold_chars

    def push(self, delta: str) -> str:
        self._buf += delta
        safe_end = self._safe_prefix_len(self._buf)
        if safe_end <= 0:
            if len(self._buf) > self.max_hold_chars:
                # The construct never closed (truncated reply, stray bracket).
                # Releasing beats holding the rest of the turn hostage.
                out, self._buf = self._buf, ""
                return strip_markdown(out)
            return ""
        out, self._buf = self._buf[:safe_end], self._buf[safe_end:]
        return strip_markdown(out)

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return strip_markdown(out) if out else ""

    @staticmethod
    def _safe_prefix_len(buf: str) -> int:
        """Longest prefix that cannot be changed by text yet to arrive.

        Complete constructs are masked out first. Looking only at the tail
        after the *last* marker is not enough: with "**bold**" arriving in
        four deltas, the last "**" closes the first one, and a rfind-based
        check would happily release the opening stars.
        """
        masked = _CODE_FENCE.sub(lambda m: "\0" * len(m.group(0)), buf)
        masked = _IMAGE.sub(lambda m: "\0" * len(m.group(0)), masked)
        masked = _LINK.sub(lambda m: "\0" * len(m.group(0)), masked)
        masked = _INLINE_CODE.sub(lambda m: "\0" * len(m.group(0)), masked)
        masked = _BOLD_ITALIC.sub(lambda m: "\0" * len(m.group(0)), masked)
        positions = [masked.find(marker) for marker in _MARKERS]
        live = [p for p in positions if p != -1]
        return min(live) if live else len(buf)
