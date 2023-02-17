import numpy as np
import random_data_generator as rdg


def normal_equation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_c = np.c_[np.ones((len(x), 1)), x]
    return np.linalg.inv(x_c.T @ x_c) @ x_c.T @ y


def main():
    m = 150
    x = rdg.generate_random_x(m)
    y = rdg.generate_random_y_from_x(x)
    theta_final = normal_equation(x, y)
    print(theta_final)


if __name__ == "__main__":
    main()
