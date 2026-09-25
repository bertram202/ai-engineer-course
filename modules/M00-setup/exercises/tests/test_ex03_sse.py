import json

import pytest
from ex03_sse import collect_stream, parse_sse


def sse(*events):
    lines = []
    for e in events:
        lines += [e if isinstance(e, str) else "data: " + json.dumps(e, ensure_ascii=False), ""]
    return lines


def chunk(content=None, **delta):
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta}]}


def test_parse_simple_stream():
    events = list(parse_sse(sse(chunk("При"), chunk("вет"), "data: [DONE]")))
    assert events == [chunk("При"), chunk("вет")]


def test_parse_stops_at_done():
    events = list(parse_sse(sse(chunk("a"), "data: [DONE]", chunk("не должно попасть"))))
    assert events == [chunk("a")]


def test_parse_skips_comments_and_blank_lines():
    lines = [": keep-alive", "", "data: " + json.dumps(chunk("x")), "", "", "data: [DONE]"]
    assert list(parse_sse(lines)) == [chunk("x")]


def test_parse_without_space_after_colon():
    assert list(parse_sse(['data:{"a":1}', "data:[DONE]"])) == [{"a": 1}]


def test_parse_is_lazy():
    def lines():
        yield "data: " + json.dumps(chunk("a"))
        raise AssertionError("parse_sse читает дальше, чем нужно")

    assert next(parse_sse(lines())) == chunk("a")


def test_collect_content():
    assert collect_stream([chunk("При"), chunk("вет")]) == ("Привет", "")


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_collect_reasoning_from_both_servers(field):
    events = [chunk(**{field: "Думаю"}), chunk(**{field: "..."}), chunk("Ответ")]
    assert collect_stream(events) == ("Ответ", "Думаю...")


def test_collect_tolerates_none_and_usage_chunk():
    events = [
        chunk(None, role="assistant"),
        chunk("ok", content_extra=None),
        {"choices": [], "usage": {"total_tokens": 3}},
    ]
    assert collect_stream(events) == ("ok", "")
