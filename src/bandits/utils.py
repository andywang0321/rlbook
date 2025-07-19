def argmax(x: list[float]) -> list[int]:
    max_val = max(x)
    return [i for i, val in enumerate(x) if val == max_val]
