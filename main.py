import numpy as np
from sklearn.base import BaseEstimator


class MyLinearSVC(BaseEstimator):
    C: int
    eta0: int
    n_epochs: int
    eta_d: int
    random_state: None | int
    Js: list[float]

    def __init__(self, C=1, eta0=1, n_epochs=1_000, eta_d=10_000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.eta_d = eta_d
        self.random_state = random_state

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)

    def fit(self, X, y):
        if self.random_state:
            np.random.seed(self.random_state)

        w = np.random.randn(X.shape[1], 1)
        b = 0

        t = np.array(y, dtype=np.float64).reshape(-1, 1) * 2 - 1

        X_t = X * t

        self.Js = []  # cost function log

        for epoch in range(self.n_epochs):
            support_vectors_idx = (X_t.dot(w) + t * b <= 1).ravel()
            X_t_sv = X_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            J: float = (1 / 2) * (w * w).sum() + self.C * ((1 - X_t_sv.dot(w) - b * t_sv).sum())
            self.Js.append(J)

            w_gradient_vector = w - self.C * X_t_sv.sum(axis=0).reshape(-1, 1)
            b_derivative = -self.C + t_sv.sum()

            eta = self.eta(epoch)
            w -= eta * w_gradient_vector
            b -= eta * b_derivative

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])

        support_vectors_idx = (X_t.dot(w) + t * b <= 1).ravel()
        self.support_vectors = X[support_vectors_idx]

        return self

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X):
        return self.decision_function(X) >= 0
