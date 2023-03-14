from sklearn import datasets
from sklearn.linear_model import LogisticRegression  # type: ignore
import matplotlib.pyplot as plt
import numpy as np


def sigmoid_array(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def main():
    example_library()
    example_manual()
    plt.show()


def example_library():
    iris = datasets.load_iris()
    iris_target: np.ndarray = iris["target"]  # type: ignore
    X: np.ndarray = iris["data"][:, 3:]  # type: ignore
    y: np.ndarray = (iris_target == 2).astype(int).reshape(len(iris_target), 1)

    model = LogisticRegression(C=10**10)
    model.fit(X, y.ravel())

    X_new = np.linspace(0, 3, 1_000).reshape(-1, 1)
    y_proba = model.predict_proba(X_new)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

    plt.subplot(2, 1, 1)
    plt.plot(X[y == 0], y[y == 0], "bs")
    plt.plot(X[y == 1], y[y == 1], "g^")
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:")
    plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")
    plt.plot(X_new, 1 - y_proba[:, 1], "b--", label="Not iris virginica")
    plt.xlabel("X")
    plt.xlabel("Probability")
    plt.axis([0, 3, -0.02, 1.02])
    plt.legend()
    plt.title("Example using library function")
    #  plt.show()


def example_manual():
    iris = datasets.load_iris()
    iris_target: np.ndarray = iris["target"]  # type: ignore
    X: np.ndarray = iris["data"][:, 3:]  # type: ignore
    X_c: np.ndarray = np.append(np.ones((len(iris["target"]), 1)), X, axis=1)  # type: ignore
    y: np.ndarray = (iris_target == 2).astype(int).reshape(len(iris_target), 1)

    eta = 0.5
    n_iterations = 100_000
    m = len(y)
    theta = np.random.randn(2, 1)

    for _ in range(n_iterations):
        gradients = (1 / m) * (X_c.T @ (sigmoid_array(X_c @ theta) - y))
        theta -= eta * gradients

    print("=== theta ===")
    print(theta)
    print()

    X_new = np.linspace(0, 3, 1_000).reshape(-1, 1)
    X_new_c = np.append(np.ones((1_000, 1)), X_new, axis=1)
    y_proba = sigmoid_array(theta.T @ X_new_c.T)
    decision_boundary = X_new[y_proba.reshape(-1, 1) >= 0.5][0]

    plt.subplot(2, 1, 2)
    plt.plot(X[y == 0], y[y == 0], "bs")
    plt.plot(X[y == 1], y[y == 1], "g^")
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:")
    plt.plot(X_new, y_proba.T, "g-", label="Iris virginica")
    plt.plot(X_new, 1 - y_proba.T, "b--", label="Not iris virginica")
    plt.xlabel("X")
    plt.xlabel("Probability")
    plt.axis([0, 3, -0.02, 1.02])
    plt.legend()
    plt.title("Example using gradient descent manually")
    #  plt.show()


if __name__ == "__main__":
    main()
