import pytest

from llm.json_utils import JSONExtractError, extract_json


def test_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fence_and_prose():
    assert extract_json('好的：\n```json\n{"a": "x"}\n```\n以上。') == {"a": "x"}


def test_think_block_removed():
    assert extract_json('<think>先想想 {"no": 1}</think>{"a": 2}') == {"a": 2}


def test_braces_inside_strings():
    assert extract_json('说明 {"speech": "她说「{别}」", "n": 1} 结束') == {"speech": "她说「{别}」", "n": 1}


def test_trailing_comma():
    assert extract_json('{"a": [1, 2,], }') == {"a": [1, 2]}


def test_truncated():
    with pytest.raises(JSONExtractError, match="截断"):
        extract_json('{"speech": "话说到一半')


def test_native_call_drift():
    with pytest.raises(JSONExtractError, match="call:respond"):
        extract_json('call:respond{speech:<|"|>你好<|"|>}')


def test_empty():
    with pytest.raises(JSONExtractError):
        extract_json("   ")
