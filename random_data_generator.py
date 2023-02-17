import numpy as np


def generate_random_x(m: int, a: int = 2) -> np.ndarray:
    return a * np.random.rand(m, 1)


def generate_random_y_from_x(
    x: np.ndarray, a: float = 3, b: float = 4, c: float = 1.5
) -> np.ndarray:
    return a + b * x + c * np.random.randn(x.size, 1)
