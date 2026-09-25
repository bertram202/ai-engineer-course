"""Упражнение 1. Сколько памяти занимает модель.

Теория: раздел «Скорость и память» в theory.pdf. 1 ГБ = 10**9 байт.
"""

GB = 10**9


def weights_gb(n_params: float, bits_per_weight: float) -> float:
    """Размер весов модели в гигабайтах.

    >>> weights_gb(8e9, 16)
    16.0
    """
    return n_params * bits_per_weight / 8 / GB


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    n_tokens: int,
    bytes_per_value: int = 2,
) -> float:
    """Размер KV-кэша в гигабайтах: ключи и значения для каждого токена в каждом слое."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_value * n_tokens / GB


def fits_in_memory(weights: float, kv_cache: float, total_ram_gb: float, reserved_gb: float = 4.0) -> bool:
    """Поместятся ли веса и KV-кэш, если системе и программам нужно `reserved_gb`."""
    return weights + kv_cache <= total_ram_gb - reserved_gb
