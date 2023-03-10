import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-z))


def main():
    x_values = np.array(
        [0.052, 0.081, 0.096, 0.158, 0.242, 0.253, 0.396, 0.427, 0.459, 0.572]
    )
    y_values = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    m = x_values.size

    theta_1 = np.arange(-40, 50, 0.15)

    h = np.array([-0.5 + theta * x_values for theta in theta_1])
    h = sigmoid(h)

    #  Función costo
    #  j = (1 / (2 * m)) * ((h - y_values) ** 2).sum(axis=1)
    j = -(1 / m) * (y_values * np.log(h) + (1 - y_values) * np.log(1 - h)).sum(axis=1)

    plt.plot(theta_1, j)
    plt.show()


if __name__ == "__main__":
    main()
