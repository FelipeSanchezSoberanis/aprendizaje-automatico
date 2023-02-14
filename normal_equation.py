import numpy as np


def normal_equation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_c = np.c_[np.ones((len(x), 1)), x]
    return np.linalg.inv(x_c.T @ x_c) @ x_c.T @ y


def main():
    m = 150
    x = 2 * np.random.rand(m, 1)
    y = 3 + 4 * x + 1.5 * np.random.randn(m, 1)
    theta_final = normal_equation(x, y)
    print(theta_final)


if __name__ == "__main__":
    main()
