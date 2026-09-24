"""Công cụ "search" mà Speech agent gọi — và trả về NGAY lập tức.

Đây là mấu chốt của sơ đồ. Một tool thường chạy xong rồi mới trả lời; tool này
chỉ *gửi* yêu cầu sang tác nhân tìm kiếm rồi trả về tức thì, kèm một lời nhắc
cho model: hãy nói với người dùng là đang tra. Nhờ vậy Speech agent tiếp tục
nói chuyện bình thường trong lúc backend làm việc, thay vì đứng im chờ.

Kết quả thật không đi qua đây. Nó về sau, và được phát thành một lượt riêng.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..base import TaskContext, ToolResult, ToolSpec
from ..search import SearchRequest

Dispatch = Callable[[str], "SearchRequest | None"]


class SearchTool:
    def __init__(self, dispatch: Dispatch, instant_ack: str = "") -> None:
        self._dispatch = dispatch
        self.instant_ack = instant_ack
        self.spec = ToolSpec(
            name="search",
            description=(
                "Gửi yêu cầu tra cứu thông tin mà bạn không tự biết chắc: dữ "
                "liệu thực tế, tin tức, số liệu, thông tin nội bộ. Trả về ngay; "
                "kết quả sẽ tới sau."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Câu hỏi cần tra, viết đầy đủ và rõ nghĩa.",
                    }
                },
                "required": ["query"],
            },
            timeout_ms=300,
        )

    async def run(self, arguments: dict[str, Any], ctx: TaskContext) -> ToolResult:
        query = str(arguments.get("query") or ctx.user_text).strip()
        if not query:
            return ToolResult(ok=False, content="Không rõ cần tra cứu gì.", error="empty_query")
        request = self._dispatch(query)
        if request is None:
            return ToolResult(
                ok=True,
                content=(
                    "Đang có quá nhiều việc tra cứu cùng lúc. Hãy nói với người "
                    "dùng rằng bạn sẽ trả lời câu hỏi trước đã."
                ),
                data={"accepted": False},
            )
        # Một câu XIN CHỜ, không phải một câu hỏi. Bản trước viết "hoặc hỏi
        # thêm chi tiết, hoặc nói điều gì hữu ích trong lúc chờ" và đo được
        # (n=12, câu cần tra cứu): xin người dùng chờ 1/12, hỏi ngược lại họ
        # 5/12. Trong hội thoại bằng giọng nói, hỏi ngược lại lúc này là hỏng:
        # người dùng bắt đầu trả lời thì lượt trả kết quả ập tới cắt ngang họ.
        if self.instant_ack:
            content = (
                f"Đã gửi yêu cầu tra cứu “{query}”. Người dùng vừa nghe bạn nói "
                "sẽ đi tra. Bây giờ nói đúng MỘT câu ngắn xin họ chờ trong giây "
                "lát. TUYỆT ĐỐI không đặt câu hỏi ngược lại: họ đang đợi, và câu "
                "hỏi của bạn sẽ bị cắt ngang khi kết quả về. Cũng không được bịa "
                "nội dung kết quả."
            )
        else:
            content = (
                f"Đã gửi yêu cầu tra cứu “{query}”. Nói đúng MỘT câu ngắn cho "
                "người dùng biết bạn đang tra và xin họ chờ trong giây lát. "
                "TUYỆT ĐỐI không đặt câu hỏi ngược lại, và không bịa nội dung "
                "kết quả."
            )
        return ToolResult(
            ok=True,
            content=content,
            data={
                "accepted": True,
                "request_id": request.id,
                # Cơ chế chung: tool nào muốn nói ngay thì đặt khoá này.
                **({"speak_now": self.instant_ack} if self.instant_ack else {}),
            },
        )
