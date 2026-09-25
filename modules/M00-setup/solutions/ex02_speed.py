"""Упражнение 2. Скорость генерации.

Теория: разделы «Две фазы генерации» и «Почему decode упирается в память».
"""


def decode_speed_limit(bandwidth_gb_s: float, model_size_gb: float) -> float:
    """Верхняя граница скорости генерации (токенов в секунду).

    Для каждого токена нужно прочитать из памяти все веса модели.
    """
    return bandwidth_gb_s / model_size_gb


def generation_speed(n_tokens: int, total_s: float, ttft_s: float) -> float:
    """Скорость фазы decode: токены ответа, делённые на время после первого токена.

    Если `total_s <= ttft_s`, измерение бессмысленно — выбросьте ValueError.
    """
    if total_s <= ttft_s:
        raise ValueError("общее время должно быть больше времени до первого токена")
    return n_tokens / (total_s - ttft_s)


def max_context_tokens(free_memory_gb: float, kv_bytes_per_token: int) -> int:
    """Сколько токенов контекста поместится в свободную память (целое, округление вниз)."""
    return int(free_memory_gb * 10**9 // kv_bytes_per_token)
