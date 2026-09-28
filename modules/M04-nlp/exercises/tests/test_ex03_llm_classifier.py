import json

import pytest
from ex03_llm_classifier import build_messages, classify, parse_label, per_class_recall, stars_to_label

from mlcourse.testing import fake_client


def test_build_messages_zero_shot():
    msgs = build_messages("Отличное платье", "Ты классификатор.")
    assert msgs == [{"role": "system", "content": "Ты классификатор."}, {"role": "user", "content": "Отличное платье"}]


def test_build_messages_few_shot():
    msgs = build_messages("текст", "система", [("плохо", "negative"), ("супер", "positive")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"]) == {"label": "negative"}
    assert msgs[-1]["content"] == "текст"


@pytest.mark.parametrize(("content", "expected"), [
    ('{"label": "neutral"}', "neutral"),
    ("positive", "positive"),
    ("Ответ: Negative.", "negative"),
])
def test_parse_label(content, expected):
    assert parse_label(content) == expected


@pytest.mark.parametrize("content", ['{"label": "angry"}', "не знаю", "positive или negative"])
def test_parse_label_rejects_unclear(content):
    with pytest.raises(ValueError):
        parse_label(content)


def test_stars_to_label():
    assert [stars_to_label(s) for s in range(1, 6)] == ["negative", "negative", "neutral", "positive", "positive"]
    with pytest.raises(ValueError):
        stars_to_label(0)


def test_classify_sends_prompt_and_parses_answer():
    client, requests = fake_client(['{"label": "positive"}'])
    label = classify(client, "fake-model", "Супер!", "система", [("ужас", "negative")])
    assert label == "positive"
    body = requests[0]
    assert body["model"] == "fake-model"
    assert body["temperature"] == 0
    assert body["messages"][-1] == {"role": "user", "content": "Супер!"}
    assert len(body["messages"]) == 4


def test_per_class_recall():
    y_true = ["negative", "negative", "neutral", "positive"]
    y_pred = ["negative", "neutral", "negative", "positive"]
    assert per_class_recall(y_true, y_pred) == {"negative": 0.5, "neutral": 0.0, "positive": 1.0}
    assert per_class_recall(["negative"], ["negative"])["neutral"] == 0.0
