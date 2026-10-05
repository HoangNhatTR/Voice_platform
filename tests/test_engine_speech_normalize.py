"""Decimals with a unit are left alone — never half-expanded.

The unit rule matched the digits AFTER the separator: "0,5%" became
"0,năm phần trăm", which a talker reads "không, năm phần trăm". Whether
"2.5" is a decimal or a thousands group is ambiguous in Vietnamese text, so
like a bare number it is the talker's to read.
"""

from __future__ import annotations

import pytest

from voiceplatform.conversation.speech_normalize import normalize_for_speech


@pytest.mark.parametrize("text", [
    "Lãi suất 0,5% một năm.",
    "Lãi 5,5%/năm",
    "Nặng 1,5 kg.",
    "Dài 2.5 km.",
    "Nhiệt độ 36,6°C",
    "Phí 1.500,5 đồng",
])
def test_a_decimal_is_left_to_the_talker(text):
    assert normalize_for_speech(text) == text


def test_whole_numbers_with_units_are_still_expanded():
    assert normalize_for_speech("Phí 1.500.000 đồng.") == "Phí một triệu năm trăm nghìn đồng."
    assert normalize_for_speech("Lãi 5% một năm.") == "Lãi năm phần trăm một năm."
    assert normalize_for_speech("Đi 12 km") == "Đi mười hai ki lô mét"
