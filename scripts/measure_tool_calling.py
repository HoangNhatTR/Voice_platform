"""Đo tỉ lệ gọi công cụ của một prompt, thay vì đoán bằng cảm giác.

Sửa prompt để "model chịu gọi tool" là loại thay đổi không thể tin bằng mắt:
cùng hai khối chữ, đảo thứ tự là từ 0/10 thành 10/10. Script này chạy hai bộ
câu hỏi — bộ CẦN công cụ và bộ KHÔNG cần — rồi in tỉ lệ gọi đúng và gọi thừa.
Đổi chữ trong system prompt thì chạy lại trước khi commit.

    PYTHONPATH=src .venv/bin/python scripts/measure_tool_calling.py \
        --endpoint http://127.0.0.1:8088/v1 --model qwen3.5-9b
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voiceplatform.conversation.context import ConversationContext
from voiceplatform.core.config import Config
from voiceplatform.models.base import Message
from voiceplatform.models.llm.openai_compat import OpenAiCompatLlm
from voiceplatform.tasks.registry import build_registry

# Bộ câu hỏi bám theo công cụ đang bật. Thêm tool thì thêm câu vào đây, nếu
# không thì con số đo được chỉ nói về cái đồng hồ.
NEEDS_TOOL = [
    "Bây giờ là mấy giờ rồi?",
    "Mấy giờ rồi bạn?",
    "Cho tôi hỏi giờ hiện tại",
    "Hôm nay là thứ mấy?",
    "Hôm nay ngày bao nhiêu?",
    "Giờ này là mấy giờ nhỉ",
    "Bạn xem giúp tôi mấy giờ",
    "Ngày tháng hôm nay thế nào?",
    "Bây giờ mấy giờ ở đây",
    "Cho tôi biết thời gian hiện tại",
]

NO_TOOL = [
    "Chào bạn",
    "Bạn tên gì?",
    "Kể tôi nghe một câu chuyện ngắn",
    "Thủ đô Việt Nam là gì?",
    "Bạn giúp được gì cho tôi?",
    "Cảm ơn bạn nhé",
    "Bạn có khỏe không?",
    "Hai cộng hai bằng mấy?",
    "Tôi muốn nhắc lại lịch hẹn vào ngày 30 tháng 9 năm 2026.",
    "Đọc lại ngày 25 tháng 12 năm 2025 giúp tôi.",
]


async def called_tool(llm: OpenAiCompatLlm, system: str, question: str, tools) -> bool:
    messages = [Message(role="system", content=system), Message(role="user", content=question)]
    async for delta in llm.stream(messages, tools=tools, max_tokens=64):
        if delta.tool_call is not None:
            return True
    return False


async def main(args: argparse.Namespace) -> int:
    config = Config.load(args.config) if args.config else Config()
    registry = build_registry(config.tasks.tools)
    tools = registry.openai_tools()
    context = ConversationContext(config.conversation.system_prompt, tool_instruction=config.conversation.tool_instruction)
    system = context.messages(tool_names=registry.names())[0].content

    print("công cụ :", registry.names())
    print("system  :", system.replace("\n", " ⏎ ")[:200])
    options = dict(config.models.llm.options) if config.models.llm.backend == "openai_compat" else {}
    options.update(endpoint=args.endpoint, model=args.model, enable_thinking=False)
    llm = OpenAiCompatLlm(**options)
    try:
        hit = 0
        misses = []
        for q in NEEDS_TOOL:
            called = await called_tool(llm, system, q, tools)
            hit += called
            if not called:
                misses.append(q)
        false = 0
        extras = []
        for q in NO_TOOL:
            called = await called_tool(llm, system, q, tools)
            false += called
            if called:
                extras.append(q)
    finally:
        await llm.close()

    print(f"\ngọi đúng : {hit}/{len(NEEDS_TOOL)}")
    print(f"gọi thừa : {false}/{len(NO_TOOL)}")
    if misses:
        print("bỏ sót   :", misses)
    if extras:
        print("gọi thừa ở:", extras)
    ok = hit >= 0.8 * len(NEEDS_TOOL) and false <= 0.1 * len(NO_TOOL)
    print("ĐẠT" if ok else "KHÔNG ĐẠT (cần ≥80% gọi đúng, ≤10% gọi thừa)")
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/local-cpu.yaml")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="qwen3.5-9b")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
