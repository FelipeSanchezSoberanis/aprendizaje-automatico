import numpy as np
import matplotlib.pyplot as plt


def main():
    m = 10

    x_array = np.linspace(-2, 2, m)
    y_array = 2.5 + 5 * x_array

    theta_0 = np.linspace(-12.5, 17.5, 50)
    theta_1 = np.linspace(-10, 20, 50)

    h = np.array([[th0 + th1 * x_array for th0 in theta_0] for th1 in theta_1])
    j = (1 / (2 * m)) * ((h - y_array) ** 2).sum(axis=2)

    plt.figure()
    ax = plt.axes(projection="3d")
    th0, th1 = np.meshgrid(theta_0, theta_1)
    ax.plot_surface(th0, th1, j, cmap="coolwarm")

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel(r"$J(\theta_0, \theta_1)$")

    plt.show()


if __name__ == "__main__":
    main()
