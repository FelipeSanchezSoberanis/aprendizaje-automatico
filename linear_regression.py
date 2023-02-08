import numpy as np
import matplotlib.pyplot as plt


def linear_regression(
    x_points: np.ndarray, y_points: np.ndarray
) -> tuple[float, float]:
    x_bar = np.average(x_points)
    y_bar = np.average(y_points)

    b_hat_top = np.sum((x_points - x_bar) * (y_points - y_bar))
    b_hat_bottom = np.sum((x_points - x_bar) ** 2)

    b_hat = b_hat_top / b_hat_bottom

    a_hat = y_bar - b_hat * x_bar

    return float(a_hat), float(b_hat)


def main():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + 1 * np.random.randn(11)
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + 1 * np.random.randn(11)

    a, b = linear_regression(x, y)

    x_linear_regression = np.linspace(min(x), max(x))
    y_linear_regression = a + b * x_linear_regression

    plt.plot(x, y, "b.")
    plt.plot(x_linear_regression, y_linear_regression, "r-")
    plt.show()


if __name__ == "__main__":
    main()
