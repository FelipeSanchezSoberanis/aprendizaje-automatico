import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import logging

logging.basicConfig(level=logging.INFO)
#  logging.disable()


def one_hot_encoder(target: np.ndarray) -> np.ndarray:
    n_classes: int = np.unique(target).shape[0]
    y_encode: np.ndarray = np.zeros((target.shape[0], n_classes))
    for idx, val in enumerate(target):
        y_encode[idx, val] = 1.0
    return y_encode


def sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def model_fit(
    data: np.ndarray, target: np.ndarray, eta: float = 0.55, iterations: int = 100000
) -> np.ndarray:
    m = len(target)
    logging.info(f"target: {m}")

    theta = np.random.randn(data.shape[1], target.shape[1])

    logging.info(f"theta: {theta}")

    for _ in range(iterations):
        gradients = (1 / m) * (data.T @ (sigmoid(data @ theta) - target))
        theta = theta - eta * gradients

    logging.info(f"theta: {theta}")

    return theta


def model_test(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    weights: np.ndarray,
) -> list[int]:
    list1 = [0, 0, 0]

    for i in range(len(list1)):
        a0 = weights.T[i][0]
        a1 = weights.T[i][1]
        a2 = weights.T[i][2]
        a3 = weights.T[i][3]
        a4 = weights.T[i][4]
        list1[i] = np.exp(
            a0 + a1 * sepal_length + a2 * sepal_width + a3 * petal_length + a4 * petal_width
        )

    maxP = np.argmax([z / sum(list1) for z in list1])

    pred = [0, 0, 0]

    pred[maxP] = 1

    return pred


def model_predict(
    data: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    predict_list = []
    test_list = []
    for i in data:
        predict_list.append(np.argmax(model_test(i[0], i[1], i[2], i[3], weights)))
    for j in target:
        test_list.append(np.argmax(j))
    num = 0
    for k in range(len(predict_list)):
        if predict_list[k] == test_list[k]:
            num = num + 1

    final_list: np.ndarray = np.array([predict_list, test_list], ndmin=2)
    effi = num / len(predict_list)

    return final_list, effi


def main():
    iris: Bunch = datasets.load_iris()  # type: ignore

    x_data: np.ndarray = iris["data"]  # type: ignore
    y_data: np.ndarray = iris["target"]  # type: ignore

    x_c: np.ndarray = np.c_[np.ones((len(x_data), 1)), x_data]
    y_c = one_hot_encoder(y_data)

    train_test_data: list[np.ndarray] = train_test_split(x_c, y_c, train_size=0.35)  # type: ignore
    x_train, x_test, y_train, y_test = train_test_data

    logging.info(f"x_train shape: {x_train.shape}")

    a = model_fit(x_train, y_train)

    predictions, efficiency = model_predict(x_test, y_test, a)

    logging.info(f"Predictions: {predictions}")
    logging.info(f"Efficiency: {efficiency}")


if __name__ == "__main__":
    main()
