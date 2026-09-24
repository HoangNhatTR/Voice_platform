from __future__ import annotations

import time
from typing import Any

from ..base import TaskContext, ToolResult, ToolSpec

_WEEKDAYS = [
    "thứ hai", "thứ ba", "thứ tư", "thứ năm",
    "thứ sáu", "thứ bảy", "chủ nhật",
]


class ClockTool:
    """Local date and time, phrased for speech rather than for a screen."""

    spec = ToolSpec(
        name="clock",
        description="Trả về ngày giờ hiện tại theo giờ địa phương.",
        parameters={"type": "object", "properties": {}},
        timeout_ms=200,
    )

    async def run(self, arguments: dict[str, Any], ctx: TaskContext) -> ToolResult:
        t = time.localtime()
        spoken = (
            f"bây giờ là {t.tm_hour} giờ {t.tm_min} phút, "
            f"{_WEEKDAYS[t.tm_wday]} ngày {t.tm_mday} tháng {t.tm_mon} năm {t.tm_year}"
        )
        return ToolResult(
            ok=True,
            content=spoken,
            data={"iso": time.strftime("%Y-%m-%dT%H:%M:%S", t)},
        )
