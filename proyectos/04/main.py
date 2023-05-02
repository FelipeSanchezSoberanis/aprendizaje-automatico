import matplotlib.pyplot as plt
import random
import math
import matplotlib.colors as mcolors


def generate_random_point_cloud() -> tuple[list[float], list[float]]:
    x_1, y_1, x_2, y_2 = 0, 0, 10, 10
    no_points: int = 200
    x_values: list[float] = []
    y_values: list[float] = []
    max_len: float = 8

    for _ in range(no_points):
        theta: float = random.randint(0, 361)
        len = random.random() * max_len
        x_values.append(x_1 + len * math.cos(math.radians(theta)))
        y_values.append(y_1 + len * math.sin(math.radians(theta)))

        theta: float = random.randint(0, 361)
        len = random.random() * max_len
        x_values.append(x_2 + len * math.cos(math.radians(theta)))
        y_values.append(y_2 + len * math.sin(math.radians(theta)))

    return x_values, y_values


def generate_random_centroids(
    x_values: list[float], y_values: list[float], k: int
) -> list[tuple[float, float]]:
    centroids: list[tuple[float, float]] = []
    random_indices: list[int] = random.sample([i for i in range(len(x_values))], k)
    for i in random_indices:
        centroids.append((x_values[i], y_values[i]))
    return centroids


def plot_point_cloud_with_centroids(
    x_values: list[float],
    y_values: list[float],
    centroids: list[tuple[float, float]],
    centroids_colors: dict[tuple[float, float], str],
    centroids_points: dict[tuple[float, float], tuple[float, float]],
):
    for x, y in zip(x_values, y_values):
        plt.scatter(x, y, c=centroids_colors[centroids_points[x, y]])
    for x, y in centroids:
        plt.scatter(x, y, c=centroids_colors[(x, y)], marker="+", s=5_000)
    plt.show()


def generate_random_color_for_centroids(
    centroids: list[tuple[float, float]]
) -> dict[tuple[float, float], str]:
    colors = random.sample(list(mcolors.TABLEAU_COLORS.keys()), len(centroids))
    centroids_colors: dict[tuple[float, float], str] = {}
    for centroid, color in zip(centroids, colors):
        centroids_colors[centroid] = color
    return centroids_colors


def get_closest_centroid_for_points(
    x_values: list[float],
    y_values: list[float],
    centroids: list[tuple[float, float]],
):
    centroids_points: dict[tuple[float, float], tuple[float, float]] = {}

    for x, y in zip(x_values, y_values):
        distances: dict[tuple[float, float], float] = {}
        for centroid in centroids:
            x_centroid, y_centroid = centroid
            distances[centroid] = math.sqrt((x_centroid - x) ** 2 + (y_centroid - y) ** 2)
        closest_centroid = min(distances, key=lambda x: distances[x])
        centroids_points[(x, y)] = closest_centroid
    return centroids_points


def main():
    x_values, y_values = generate_random_point_cloud()
    centroids = generate_random_centroids(x_values, y_values, 2)
    centroids_colors = generate_random_color_for_centroids(centroids)
    centroids_points = get_closest_centroid_for_points(x_values, y_values, centroids)
    plot_point_cloud_with_centroids(
        x_values, y_values, centroids, centroids_colors, centroids_points
    )


if __name__ == "__main__":
    main()
