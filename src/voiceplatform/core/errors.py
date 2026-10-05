"""Error taxonomy.

A voice session must degrade rather than die: every failure below is caught by
the engine, reported on the wire, and answered with a spoken fallback.
"""

from __future__ import annotations


class VoicePlatformError(Exception):
    """Base class."""


class ConfigError(VoicePlatformError):
    pass


class TransportError(VoicePlatformError):
    pass


class ModelUnavailable(VoicePlatformError):
    """The engine is not loaded / the endpoint is unreachable."""


class ModelTimeout(VoicePlatformError):
    """The engine took longer than its budget."""


class CapacityExceeded(ModelUnavailable):
    """The process has reached its bounded work capacity."""


class ToolError(VoicePlatformError):
    pass


class ToolTimeout(ToolError):
    pass


class IllegalTransition(VoicePlatformError):
    """The turn state machine was asked for a transition it does not allow."""
