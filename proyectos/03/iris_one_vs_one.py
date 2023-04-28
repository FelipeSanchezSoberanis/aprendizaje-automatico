from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logging.disable()


def one_vs_one_filter(value: float, option_1: float, option_2: float) -> int:
    if value == option_1:
        return 1
    elif value == option_2:
        return 0
    return -1


def create_combinations(targets: dict[int, str]) -> list[tuple[int, int]]:
    combinations: list[tuple[int, int]] = []
    keys = targets.keys()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            combinations.append((i, j))
    return combinations


def main():
    iris: Bunch = load_iris()  # type: ignore

    x_data: np.ndarray = iris.get("data")  # type: ignore
    y_data: np.ndarray = iris.get("target")  # type: ignore

    target_names: list[str] = iris.get("target_names")  # type:ignore
    targets: dict[int, str] = {k: v for k, v in enumerate(target_names)}
    combinations = create_combinations(targets)

    logging.info(f"Combinations: {combinations}")

    train_test_arrays: list[np.ndarray] = train_test_split(
        x_data, y_data
    )  # type:ignore
    x_train, x_test, y_train, y_test = train_test_arrays

    models: dict[tuple[int, int], LogisticRegression] = {}

    for combination in combinations:
        class_1, class_2 = combination

        logging.info(f"Training model for {targets[class_1]} vs {targets[class_2]}")

        helper_array_model_1 = np.ones((len(y_train), 1)).flatten() * class_1
        helper_array_model_2 = np.ones((len(y_train), 1)).flatten() * class_2

        logging.info(np.mean(helper_array_model_1, dtype=float))
        logging.info(np.mean(helper_array_model_2, dtype=float))

        y_train_mapped = np.array(
            list(
                map(
                    one_vs_one_filter,
                    y_train,
                    helper_array_model_1,
                    helper_array_model_2,
                )
            )
        )

        x_train_mapped = x_train[y_train_mapped != -1]
        y_train_mapped = y_train_mapped[y_train_mapped != -1]

        model = LogisticRegression(C=10**10)
        model.fit(x_train_mapped, y_train_mapped)

        logging.info(f"Classses are ordered like this: {model.classes_}")

        models[combination] = model

    logging.info(f"Models trained: {models}")

    success: list[bool] = []
    for x, y in zip(x_test, y_test):
        probabilities: dict[int, list[float]] = {}
        for combination, model in models.items():
            class_1, class_2 = combination

            logging.info(f"Class 1: {class_1}")
            logging.info(f"Class 2: {class_2}")

            results: np.ndarray = model.predict_proba(x.reshape(1, -1))
            prob_class_2 = results[0, 0]
            prob_class_1 = results[0, 1]

            if class_1 not in probabilities:
                probabilities[class_1] = []
            if class_2 not in probabilities:
                probabilities[class_2] = []

            probabilities[class_1] = probabilities[class_1] + [prob_class_1]
            probabilities[class_2] = probabilities[class_2] + [prob_class_2]

        logging.info(probabilities)

        predicted = max(
            probabilities, key=lambda x: np.mean(probabilities[x], dtype=float)
        )
        expected = y

        if predicted == expected:
            success.append(True)
        else:
            success.append(False)

        logging.info(f"Predicted: {predicted}. Expected: {expected}")

    print(f"Success rate: {success.count(True) / len(success)}")


if __name__ == "__main__":
    main()
