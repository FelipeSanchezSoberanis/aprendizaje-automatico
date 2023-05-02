import matplotlib.pyplot as plt
import random
import math
import matplotlib.colors as mcolors
import os


GIF_DIR = os.path.join("proyectos", "04", "media")
ITERATIONS = 10
CLUSTERS = 2

Coordinate = tuple[float, float]


def generate_random_point_cloud() -> tuple[list[float], list[float]]:
    x_1, y_1, x_2, y_2 = 0, 0, 10, 10
    no_points: int = 200
    x_values: list[float] = []
    y_values: list[float] = []
    max_len: float = 8

    for _ in range(no_points):
        theta = random.randint(0, 361)
        len = random.random() * max_len
        x_values.append(x_1 + len * math.cos(math.radians(theta)))
        y_values.append(y_1 + len * math.sin(math.radians(theta)))

        theta = random.randint(0, 361)
        len = random.random() * max_len
        x_values.append(x_2 + len * math.cos(math.radians(theta)))
        y_values.append(y_2 + len * math.sin(math.radians(theta)))

    return x_values, y_values


def generate_random_centroids(
    x_values: list[float], y_values: list[float], k: int
) -> list[Coordinate]:
    centroids: list[Coordinate] = []
    random_indices: list[int] = random.sample([i for i in range(len(x_values))], k)
    for i in random_indices:
        centroids.append((x_values[i], y_values[i]))
    return centroids


def plot_point_cloud_with_centroids(
    x_values: list[float],
    y_values: list[float],
    centroids: list[Coordinate],
    color_per_centroid: dict[Coordinate, str],
    closest_centroid_per_point: dict[Coordinate, Coordinate],
    iteration: int,
):
    plt.figure()
    for x, y in zip(x_values, y_values):
        closest_centroid = closest_centroid_per_point[(x, y)]
        plt.scatter(x, y, c=color_per_centroid[closest_centroid])
    for x, y in centroids:
        plt.scatter(x, y, c=color_per_centroid[(x, y)], marker="+", s=5_000)
    plt.title(f"Iteration: {iteration + 1}")
    plt.savefig(os.path.join(GIF_DIR, f"iteration-{str(iteration + 1).zfill(2)}"), dpi=300)
    print(f"Saved image {iteration + 1}/{ITERATIONS}")


def generate_random_color_per_centroid(centroids: list[Coordinate]) -> dict[Coordinate, str]:
    colors = random.sample(list(mcolors.TABLEAU_COLORS.keys()), len(centroids))
    centroids_colors: dict[Coordinate, str] = {}
    for centroid, color in zip(centroids, colors):
        centroids_colors[centroid] = color
    return centroids_colors


def get_closest_centroid_per_point(
    x_values: list[float],
    y_values: list[float],
    centroids: list[Coordinate],
):
    centroids_points: dict[Coordinate, Coordinate] = {}

    for x, y in zip(x_values, y_values):
        distances: dict[Coordinate, float] = {}
        for centroid in centroids:
            x_centroid, y_centroid = centroid
            distances[centroid] = math.sqrt((x_centroid - x) ** 2 + (y_centroid - y) ** 2)
        closest_centroid = min(distances, key=lambda x: distances[x])
        centroids_points[(x, y)] = closest_centroid
    return centroids_points


def update_centroids(
    closest_centroid_per_point: dict[Coordinate, Coordinate],
    old_centroids: list[Coordinate],
) -> list[Coordinate]:
    new_centroids: list[Coordinate] = []
    points_per_centroid: dict[Coordinate, list[Coordinate]] = {}
    for point, centroid in closest_centroid_per_point.items():
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
    centroids: list[Coordinate],
    color_per_centroid: dict[Coordinate, str],
    old_centroids: list[Coordinate],
):
    new_color_per_centroid: dict[Coordinate, str] = {}

    for new_centroid, old_centroid in zip(centroids, old_centroids):
        new_color_per_centroid[new_centroid] = color_per_centroid[old_centroid]

    return new_color_per_centroid


def main():
    x_values, y_values = generate_random_point_cloud()

    centroids = generate_random_centroids(x_values, y_values, CLUSTERS)
    color_per_centroid = generate_random_color_per_centroid(centroids)
    for i in range(ITERATIONS):
        closest_centroid_per_point = get_closest_centroid_per_point(x_values, y_values, centroids)
        plot_point_cloud_with_centroids(
            x_values, y_values, centroids, color_per_centroid, closest_centroid_per_point, i
        )

        if i == ITERATIONS - 1:
            break

        old_centroids = centroids.copy()
        centroids = update_centroids(closest_centroid_per_point, old_centroids)
        color_per_centroid = update_colors_mapping(centroids, color_per_centroid, old_centroids)


if __name__ == "__main__":
    main()
