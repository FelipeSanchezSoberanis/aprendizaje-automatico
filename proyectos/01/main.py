import matplotlib.pyplot as plt
import numpy as np
import random


def eval_poly(coeffs: np.ndarray, x: float) -> float:
    total: float = 0
    for power, coeff in enumerate(coeffs[::-1]):
        total += (x**power) * coeff
    return total


def diff_poly(coeffs: np.ndarray) -> list[float]:
    diff_coeffs: list[float] = []
    for i, coeff in enumerate(coeffs[:-1]):
        diff_coeffs.append(coeff * (len(coeffs) - (i + 1)))
    return diff_coeffs


def gradient_descent(
    x_values: np.ndarray,
    y_values: np.ndarray,
    no_weights: int,
    learning_rate: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_c = np.c_[np.ones((x_values.size, no_weights - 1)), x_values]
    weights = np.random.randn(no_weights, 1)
    error_log = np.zeros(iterations)
    for i in range(iterations):
        error_log[i] = (1 / (2 * x_values.size)) * (
            (x_c @ weights - y_values) ** 2
        ).sum()
        gradients = (1 / x_values.size) * (x_c.T @ (x_c @ weights - y_values))
        weights -= learning_rate * gradients
    return weights[::-1], error_log


def main():
    coeffs: np.ndarray = np.array([2, 15])
    init_value, end_value = -10, 10
    data_size = end_value - init_value + 1
    no_weights = coeffs.size
    iterations = 1_000
    learning_rate = 0.05

    x_values = np.linspace(init_value, end_value, data_size).reshape(data_size, 1)
    y_values = np.array(
        [eval_poly(coeffs, x) * random.randint(75, 125) / 100 for x in x_values]
    ).reshape(data_size, 1)

    result_coeffs, error_log = gradient_descent(
        x_values, y_values, no_weights, learning_rate, iterations
    )
    y_predicted = np.array([eval_poly(result_coeffs, x) for x in x_values])

    print("Expected result: {}".format(coeffs))
    print("Calculated result: {}".format(result_coeffs.reshape(1, no_weights)))

    plt.subplot(1, 2, 1)
    plt.plot(x_values, y_values, "b.", label="Input data")
    plt.plot(x_values, y_predicted, "r", label="Predicted data")
    plt.title("x vs y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(iterations), error_log)
    plt.title("iterations vs error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")

    plt.show()


if __name__ == "__main__":
    main()
