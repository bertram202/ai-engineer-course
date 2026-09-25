import pytest
from ex02_speed import decode_speed_limit, generation_speed, max_context_tokens


def test_decode_limit_m1_qwen3_8b():
    assert decode_speed_limit(68, 5.2) == pytest.approx(13.08, abs=0.01)


def test_decode_limit_grows_with_bandwidth():
    assert decode_speed_limit(1000, 20) == pytest.approx(50)


def test_generation_speed_excludes_ttft():
    # 100 токенов за 12 с, из них 2 с — ожидание первого токена
    assert generation_speed(100, total_s=12, ttft_s=2) == pytest.approx(10)


def test_generation_speed_rejects_bad_timings():
    with pytest.raises(ValueError):
        generation_speed(10, total_s=1, ttft_s=1)


def test_max_context_tokens():
    assert max_context_tokens(2.0, 147_456) == 13_563
    assert isinstance(max_context_tokens(2.0, 147_456), int)
