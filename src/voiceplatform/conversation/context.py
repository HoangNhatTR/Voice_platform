"""Conversation history.

The rule that matters: an interrupted answer is stored as what the user
actually *heard*, not as what the model generated. Storing the full generated
text after a barge-in makes the assistant refer back to sentences that were
never spoken, and the user cannot tell why it is confused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.base import Message


@dataclass(slots=True)
class Turn:
    turn_id: int
    user_text: str = ""
    assistant_text: str = ""
    spoken_text: str = ""
    interrupted: bool = False
    tool_calls: list[str] = field(default_factory=list)


class ConversationContext:
    def __init__(self, system_prompt: str, history_turns: int = 12, *, tool_instruction: str = "") -> None:
        self.system_prompt = system_prompt
        self.tool_instruction = tool_instruction
        self.history_turns = history_turns
        self.turns: list[Turn] = []
        self._pending_tool_messages: list[Message] = []

    def start_turn(self, turn_id: int, user_text: str) -> Turn:
        turn = Turn(turn_id=turn_id, user_text=user_text)
        self.turns.append(turn)
        del self.turns[:-self.history_turns]
        self._pending_tool_messages.clear()
        return turn

    def start_delivery_turn(self, turn_id: int) -> Turn:
        """Lượt do hệ thống tự mở — kết quả tra cứu về.

        Khác `start_turn`: không có lời người dùng, và KHÔNG xoá các message
        tool đang chờ (chính chúng là thứ cần đọc ra). Phải là một lượt riêng,
        nếu không câu trả lời muộn sẽ ghi đè lên câu "đang tra cứu" mà người
        dùng đã nghe, và lịch sử không còn khớp với những gì đã xảy ra.
        """
        turn = Turn(turn_id=turn_id, user_text="")
        self.turns.append(turn)
        del self.turns[:-self.history_turns]
        return turn

    def add_tool_exchange(self, tool_name: str, content: str) -> None:
        if self.turns:
            self.turns[-1].tool_calls.append(tool_name)
        self._pending_tool_messages.append(
            Message(role="tool", content=content, name=tool_name, tool_call_id=tool_name)
        )

    def commit_assistant(self, generated: str, spoken: str, interrupted: bool) -> None:
        if not self.turns:
            return
        turn = self.turns[-1]
        turn.assistant_text = generated.strip()
        turn.spoken_text = spoken.strip()
        turn.interrupted = interrupted

    def preview(self, user_text: str, *, tool_names: list[str] | None = None) -> list[Message]:
        """The messages round 0 WOULD send if `user_text` started a turn now.

        Nothing is changed: a speculative request is built from this, and the
        confirmed turn only adopts it if its real messages come out equal.
        """
        window = (self.turns + [Turn(turn_id=-1, user_text=user_text)])[-self.history_turns :]
        return self._render(window, tool_names, pending=[])

    def messages(
        self,
        *,
        tool_names: list[str] | None = None,
        include_pending_tools: bool = True,
    ) -> list[Message]:
        return self._render(
            self.turns[-self.history_turns :],
            tool_names,
            pending=self._pending_tool_messages if include_pending_tools else [],
        )

    def _render(self, window: list[Turn], tool_names: list[str] | None, *, pending: list[Message]) -> list[Message]:
        out = [Message(role="system", content=self._system(tool_names))]
        for turn in window:
            if turn.user_text:
                out.append(Message(role="user", content=turn.user_text))
            if turn.interrupted:
                # Only what was heard. Cut before a full phrase played, the
                # answer was never said: falling back to the generated text
                # here told the model it had said it.
                content = " ".join(p for p in (turn.spoken_text, "(bị người dùng ngắt lời)") if p)
                out.append(Message(role="assistant", content=content))
                continue
            heard = turn.spoken_text or turn.assistant_text
            if heard:
                out.append(Message(role="assistant", content=heard))
        out.extend(pending)
        return out

    def _system(self, tool_names: list[str] | None) -> str:
        """System prompt, with the tool instruction in front of it.

        Two findings, measured on Qwen3.5-9B against this repo's clock tool
        (scripts/measure_tool_calling.py, 10 questions that need the tool and
        8 that do not):

        * order matters enormously — the same two blocks scored 0/10 with the
          speech rules first and 10/10 with the tool rule first;
        * length matters too — a long "write like speech, at most two
          sentences, no lists" block suppresses tool calling even when it comes
          second (2/10).

        So the tool rule leads, and a voice prompt that follows it is kept
        short. False calls were 0/8 in every arrangement, so leading with the
        tool rule costs nothing. Rewording either block means re-running the
        measurement; this is not a place for taste.
        """
        if not tool_names:
            return self.system_prompt
        listed = ", ".join(sorted(tool_names))
        if self.tool_instruction:
            routes = []
            if 'clock' in tool_names:
                routes.append('Câu hỏi về giờ, ngày, tháng, năm hoặc thứ hiện tại: bắt buộc gọi clock.')
            if 'search' in tool_names:
                routes.append('Yêu cầu tra cứu hoặc nêu rõ nguồn: bắt buộc gọi search, kể cả khi đã biết câu trả lời. Thông tin mới: bắt buộc gọi search.')
            rule = self.tool_instruction.replace("{tools}", listed).replace("{tool_routes}", ' '.join(routes))
            return rule + f"\n\n{self.system_prompt}"
        return (
            f"Bạn có các công cụ sau: {listed}. Khi câu hỏi cần dữ liệu thực tế "
            "— thời gian, số liệu, hay bất cứ thứ gì phải tra cứu — hãy gọi "
            "công cụ rồi mới trả lời. Tuyệt đối không đoán những dữ liệu đó."
            f"\n\n{self.system_prompt}"
        )

    def find(self, turn_id: int) -> Turn | None:
        for turn in reversed(self.turns):
            if turn.turn_id == turn_id:
                return turn
        return None

    def drop_last(self, turn_id: int) -> bool:
        """Remove the newest turn if it is `turn_id` — and only then.

        Used when that turn is folded into the next one. Removing from the
        middle would rewrite history the model has already answered against.
        """
        if self.turns and self.turns[-1].turn_id == turn_id:
            self.turns.pop()
            return True
        return False

    def last_user_text(self) -> str:
        return self.turns[-1].user_text if self.turns else ""
