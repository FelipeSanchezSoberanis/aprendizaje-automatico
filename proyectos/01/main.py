import matplotlib.pyplot as plt
import numpy as np
import random

COEFFS: np.ndarray = np.array([2, 15])
INIT_VALUE = -10
END_VALUE = 10
ITERATIONS: int = 500
LEARNING_RATE: float = 0.05


def eval_poly(coeffs: np.ndarray, x: float) -> float:
    total: float = 0
    for power, coeff in enumerate(coeffs[::-1]):
        total += (x**power) * coeff
    return total


def gradient_descent(
    x_values: np.ndarray,
    y_values: np.ndarray,
    no_weights: int,
    learning_rate: float,
    iterations: int,
    batch_size: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size == -1:
        batch_size = x_values.size

    x_c = np.c_[np.ones((x_values.size, no_weights - 1)), x_values]
    weights = np.random.randn(no_weights, 1)
    error_log = np.zeros(iterations)

    for i in range(iterations):
        x_c_batch, y_values_batch = zip(
            *random.sample(list(zip(x_c, y_values)), batch_size)
        )

        x_c_batch = np.array(list(x_c_batch))
        y_values_batch = np.array(list(y_values_batch))

        multi = np.array(x_c_batch @ weights - y_values_batch)
        multi_squared = multi**2

        error_log[i] = (1 / (2 * batch_size)) * (multi_squared).sum()

        gradients = (1 / batch_size) * (x_c_batch.T @ multi)

        weights -= learning_rate * gradients

    return weights[::-1], error_log


def main():
    data_size = END_VALUE - INIT_VALUE + 1
    no_weights = COEFFS.size

    x_values = np.linspace(INIT_VALUE, END_VALUE, data_size).reshape(data_size, 1)
    y_values = np.array(
        [eval_poly(COEFFS, x) * random.randint(90, 110) / 100 for x in x_values]
    ).reshape(data_size, 1)

    #  === Batch ===
    batch_result_coeffs, batch_error_log = gradient_descent(
        x_values, y_values, no_weights, LEARNING_RATE, ITERATIONS
    )
    batch_y_predicted = np.array([eval_poly(batch_result_coeffs, x) for x in x_values])

    #  === Stochastic ===
    stochastic_result_coeffs, stochastic_error_log = gradient_descent(
        x_values, y_values, no_weights, LEARNING_RATE, ITERATIONS, batch_size=1
    )
    stochastic_y_predicted = np.array(
        [eval_poly(stochastic_result_coeffs, x) for x in x_values]
    )

    #  === Mini-batch ===
    mini_batch_result_coeffs, mini_batch_error_log = gradient_descent(
        x_values,
        y_values,
        no_weights,
        LEARNING_RATE,
        ITERATIONS,
        batch_size=int(x_values.size * 0.50),
    )
    mini_batch_y_predicted = np.array(
        [eval_poly(mini_batch_result_coeffs, x) for x in x_values]
    )

    print("Expected result: {}".format(COEFFS))
    print(
        "Batch calculated result: {}".format(batch_result_coeffs.reshape(1, no_weights))
    )
    print(
        "Stochastci calculated result: {}".format(
            stochastic_result_coeffs.reshape(1, no_weights)
        )
    )
    print(
        "Mini-batch calculated result: {}".format(
            mini_batch_result_coeffs.reshape(1, no_weights)
        )
    )

    fig, _ = plt.subplots(3, 2)
    fig.tight_layout()  # type: ignore

    #  === Batch ===
    plt.subplot(3, 2, 1)
    plt.plot(x_values, y_values, "b.", label="Input data")
    plt.plot(x_values, batch_y_predicted, "r", label="Predicted data")
    plt.title("Batch: x vs y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(range(ITERATIONS), batch_error_log)
    plt.title("Batch: iterations vs error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")

    #  === Stochastic ===
    plt.subplot(3, 2, 3)
    plt.plot(x_values, y_values, "b.", label="Input data")
    plt.plot(x_values, stochastic_y_predicted, "r", label="Predicted data")
    plt.title("Stochastic: x vs y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(range(ITERATIONS), stochastic_error_log)
    plt.title("Stochastic: iterations vs error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")

    #  === Mini-batch ===
    plt.subplot(3, 2, 5)
    plt.plot(x_values, y_values, "b.", label="Input data")
    plt.plot(x_values, mini_batch_y_predicted, "r", label="Predicted data")
    plt.title("Mini-batch: x vs y")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(range(ITERATIONS), mini_batch_error_log)
    plt.title("Mini-batch: iterations vs error")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()


if __name__ == "__main__":
    main()
