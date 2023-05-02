import matplotlib.pyplot as plt
import random
import math
import matplotlib.colors as mcolors
import os


GIF_DIR = os.path.join("proyectos", "04", "media")
ITERATIONS = 10


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
    i: int,
):
    plt.figure()
    for x, y in zip(x_values, y_values):
        plt.scatter(x, y, c=centroids_colors[centroids_points[x, y]])
    for x, y in centroids:
        plt.scatter(x, y, c=centroids_colors[(x, y)], marker="+", s=5_000)
    plt.title(f"Iteration: {i + 1}")
    plt.savefig(os.path.join(GIF_DIR, f"iteration-{str(i + 1).zfill(2)}"), dpi=300)
    print(f"Saved image {i + 1}/{ITERATIONS}")


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


def update_centroids(
    centroids_points: dict[tuple[float, float], tuple[float, float]],
    old_centroids: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    new_centroids: list[tuple[float, float]] = []
    points_per_centroid: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for point, centroid in centroids_points.items():
        if centroid not in points_per_centroid:
            points_per_centroid[centroid] = []
        points_per_centroid[centroid].append(point)

    for centroid in old_centroids:
        x_values: list[float] = []
        y_values: list[float] = []
        for x, y in points_per_centroid[centroid]:
            x_values.append(x)
            y_values.append(y)
        new_centroids.append((sum(x_values) / len(x_values), sum(y_values) / len(y_values)))
    return new_centroids


def update_colors_mapping(
    centroids: list[tuple[float, float]],
    centroids_colors: dict[tuple[float, float], str],
    old_centroids: list[tuple[float, float]],
):
    new_centroids_colors: dict[tuple[float, float], str] = {}

    for new_centroid, old_centroid in zip(centroids, old_centroids):
        new_centroids_colors[new_centroid] = centroids_colors[old_centroid]

    return new_centroids_colors


def main():
    x_values, y_values = generate_random_point_cloud()

    centroids = generate_random_centroids(x_values, y_values, 2)
    centroids_colors = generate_random_color_for_centroids(centroids)
    for i in range(ITERATIONS):
        centroids_points = get_closest_centroid_for_points(x_values, y_values, centroids)
        plot_point_cloud_with_centroids(
            x_values, y_values, centroids, centroids_colors, centroids_points, i
        )

        if i == ITERATIONS - 1:
            break

        old_centroids = centroids.copy()
        centroids = update_centroids(centroids_points, old_centroids)
        centroids_colors = update_colors_mapping(centroids, centroids_colors, old_centroids)


if __name__ == "__main__":
    main()
