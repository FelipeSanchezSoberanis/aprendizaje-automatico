import random_data_generator as rdg
import numpy as np


def main():
    eta = 0.01
    no_iterations = 1_000_000

    x = rdg.generate_random_x(150)
    y = rdg.generate_random_y_from_x(x)
    m = x.size

    x_c = np.c_[np.ones((len(x), 1)), x]

    theta = np.random.rand(2, 1)

    j_log = np.zeros(no_iterations)

    for i in range(no_iterations):
        j_log[i] = (1 / (2 * m)) * ((x_c @ theta - y) ** 2).sum()
        gradients = (1 / m) * (x_c.T @ (x_c @ theta - y))
        theta -= eta * gradients

    print(theta)


if __name__ == "__main__":
    main()
