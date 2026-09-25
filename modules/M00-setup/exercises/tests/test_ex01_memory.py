import pytest
from ex01_memory import fits_in_memory, kv_cache_gb, weights_gb


def test_weights_bf16():
    assert weights_gb(8e9, 16) == pytest.approx(16.0)


def test_weights_4bit():
    assert weights_gb(8e9, 4) == pytest.approx(4.0)


def test_kv_cache_qwen3_8b_per_token():
    # 2 · 36 · 8 · 128 · 2 = 147 456 байт на токен
    assert kv_cache_gb(36, 8, 128, 1) == pytest.approx(147_456 / 1e9)


def test_kv_cache_qwen3_8b_16k():
    assert kv_cache_gb(36, 8, 128, 16_384) == pytest.approx(2.416, abs=1e-3)


def test_kv_cache_fp32_is_twice_fp16():
    fp16 = kv_cache_gb(36, 8, 128, 1000, bytes_per_value=2)
    fp32 = kv_cache_gb(36, 8, 128, 1000, bytes_per_value=4)
    assert fp32 == pytest.approx(2 * fp16)


def test_fits_in_memory():
    assert fits_in_memory(5.2, 2.4, total_ram_gb=16) is True
    assert fits_in_memory(16.0, 2.4, total_ram_gb=16) is False


def test_fits_in_memory_respects_reserved():
    assert fits_in_memory(10, 2, total_ram_gb=16, reserved_gb=4) is True
    assert fits_in_memory(10, 2, total_ram_gb=16, reserved_gb=5) is False
