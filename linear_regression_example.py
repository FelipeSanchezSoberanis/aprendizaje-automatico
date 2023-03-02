import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # LIBRERÍA
from sklearn import metrics


def main():
    df = pd.read_csv("USA_Housing.csv")
    X = df[
        [
            "Avg. Area Income",
            "Avg. Area House Age",
            "Avg. Area Number of Rooms",
            "Avg. Area Number of Bedrooms",
            "Area Population",
        ]
    ]
    Y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    model.intercept_
    model.coef_
    y_predicted = model.predict(X_test)
    plt.scatter(y_predicted, y_test)
    plt.show()
    metrics.mean_absolute_error(y_test, y_predicted)
    metrics.mean_squared_error(y_test, y_predicted)
    np.sqrt(metrics.mean_squared_error(y_test, y_predicted))


if __name__ == "__main__":
    main()
