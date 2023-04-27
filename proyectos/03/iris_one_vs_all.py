from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def main():
    iris: Bunch = load_iris()  # type: ignore

    x_data: np.ndarray = iris.get("data")  # type: ignore
    y_data: np.ndarray = iris.get("target")  # type: ignore

    target_names: list[str] = iris.get("target_names")  # type:ignore
    targets: dict[int, str] = {k: v for k, v in enumerate(target_names)}

    train_test_arrays: list[np.ndarray] = train_test_split(
        x_data, y_data, train_size=0.9
    )  # type:ignore
    x_train, x_test, y_train, y_test = train_test_arrays

    one_vs_all_filter = lambda data, value: 1 if data == value else 0

    models: dict[str, LogisticRegression] = {}

    for target_value, target_name in targets.items():
        logging.info(f"Training model for {target_name}")

        helper_array = np.ones((len(y_data), 1)).flatten() * target_value
        y_train_mapped = np.array(list(map(one_vs_all_filter, y_train, helper_array)))

        model = LogisticRegression()
        model.fit(x_train, y_train_mapped)

        models[target_name] = model

    success: list[bool] = []
    for x, y in zip(x_test, y_test):
        probabilites: dict[str, float] = {}

        for model_name, model in models.items():
            probability = model.predict_proba(x.reshape(1, -1))
            probabilites[model_name] = probability[0, 0]

            logging.info(
                f"Probability that input {x} is of class {model_name}: {probability[0, 0]}"
            )

        predicted_class = max(probabilites, key=lambda x: probabilites[x])
        logging.info(f"Max probabilty is for class {predicted_class}")

        if predicted_class == targets[y]:
            success.append(True)
        else:
            success.append(False)

    print(f"Success rate: {success.count(True) / len(success)}")


if __name__ == "__main__":
    main()
