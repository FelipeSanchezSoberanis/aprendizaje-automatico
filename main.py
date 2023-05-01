from seaborn import scatterplot
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def main():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=float)
    scaler = MinMaxScaler()


if __name__ == "__main__":
    main()
