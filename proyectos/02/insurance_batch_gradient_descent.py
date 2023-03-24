import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tabulate import tabulate
from insurance_normal_equation import (
    add_ones_col,
    prepare_data,
    get_train_test_data,
    separate_data,
    PROYECT_HOME,
)


def log_results(predicted_data_y: np.ndarray, testing_data_y: np.ndarray) -> None:
    cols = ["Expected", "Calculated", "Error percentage"]
    data: list[list] = []

    for i in range(predicted_data_y.shape[0]):
        expected_value = testing_data_y[i, 0]
        calculated_value = predicted_data_y[i, 0]
        error_percentage = expected_value / calculated_value * 100

        data.append([expected_value, calculated_value, error_percentage])

    with open(
        os.path.join(PROYECT_HOME, "results", "insurance_batch_gradient_descent"), "w"
    ) as file:
        file.write(tabulate(data, headers=cols, tablefmt="grid"))


def plot_results(
    predicted_data_y: np.ndarray,
    testing_data_y: np.ndarray,
    training_data_percentage: float,
) -> None:
    plt.plot(testing_data_y, predicted_data_y, ".")
    plt.title(
        f"Batch gradient descent: correct y value vs predicted y value ({training_data_percentage*100} % as training data)"
    )
    plt.xlabel("Correct y value")
    plt.ylabel("Predicted y value")
    plt.show()


def batch_gradient_descent(
    training_data_x: np.ndarray,
    training_data_y: np.ndarray,
    iterations: int,
    learning_rate: float,
) -> np.ndarray:
    training_data_x_c = add_ones_col(training_data_x)
    theta = np.random.randn(training_data_x_c.shape[1], 1)
    n = training_data_x_c.shape[0]

    for _ in range(iterations):
        multi = training_data_x_c @ theta - training_data_y
        gradients = (((1 / n) * learning_rate) * training_data_x_c.T) @ multi
        theta -= gradients

    return theta


def main():
    csv_file = os.path.join(PROYECT_HOME, "data", "insurance.csv")
    insurance_df = pd.read_csv(csv_file)

    insurance_df = prepare_data(insurance_df)

    training_data_percentage = 0.9
    training_data, testing_data = get_train_test_data(
        insurance_df, training_data_percentage
    )

    training_data_x, training_data_y = separate_data(training_data)
    testing_data_x, testing_data_y = separate_data(testing_data)

    learning_rate = 0.0007
    iterations = 10_000

    start_time = time.perf_counter()
    theta = batch_gradient_descent(
        training_data_x, training_data_y, iterations, learning_rate
    )
    end_time = time.perf_counter()

    testing_data_x_c = add_ones_col(testing_data_x)
    predicted_data_y = testing_data_x_c @ theta

    average_error = np.sum(np.abs(predicted_data_y - testing_data_y)) / training_data_y.shape[0]  # type: ignore
    print(f"Batch gradient descent took {(end_time - start_time) * 1000} ms")
    print(
        f"Average error for batch gradient descent: {average_error}. (Learning rate: {learning_rate}. Iterations: {iterations})"
    )

    log_results(predicted_data_y, testing_data_y)
    plot_results(predicted_data_y, testing_data_y, training_data_percentage)


if __name__ == "__main__":
    main()
