import time
import numpy as np
import pandas as pd
import os
from insurance_normal_equation import (
    add_ones_col,
    prepare_data,
    get_train_test_data,
    separate_data,
    log_results,
    plot_results,
    Methods,
    PROYECT_HOME,
)


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

    log_results(predicted_data_y, testing_data_y, Methods.BATCH_GRADIENT_DESCENT)
    plot_results(
        predicted_data_y,
        testing_data_y,
        training_data_percentage,
        Methods.BATCH_GRADIENT_DESCENT,
    )


if __name__ == "__main__":
    main()
