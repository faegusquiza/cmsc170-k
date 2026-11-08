import math
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
import csv 

from read_data import MONTHS, read_monthly_data, standardize_data

OUTPUT_FOLDER = Path("results/dbscan/monthly")

# DBSCAN Settings
MINPTS = 5

class Point:
    def __init__(self, region, coordinates):
        self.region = region
        self.coordinates = coordinates
        self.is_core = False
        self.neighbors = []
        self.cluster = -1

    def __str__(self):
        coordinates = [f"{coord:.2f}" for coord in self.coordinates]

        return f"coordinates: {coordinates}, is_core: {self.is_core}, cluster: {self.cluster}"

def main():
    regions, monthly_values, monthly_parameters, monthly_averages = read_monthly_data()

    for month in MONTHS:
        print(f"Clustering {month}")

        points = [Point(regions[i], monthly_averages[month][i]) for i in range(len(regions))]
        epsilon = calculate_epsilon([point.coordinates for point in points])
        clusters_count = -1

        identify_neighbors(points, epsilon)

        core_points = identify_core_points(points, MINPTS)

        for point in core_points:
            if point.cluster != -1:
                continue

            clusters_count += 1
            cluster(point, clusters_count)

        # Create output directory if it doesn"t exist
        graph_clusters(points, month, monthly_parameters[month])
        graph_clusters_4d(points, month, monthly_parameters[month])
        save_clusters(points, month)

def save_clusters(points: list[Point], month):
    output_directory = OUTPUT_FOLDER / month
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "clusters.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["REGION", "CLUSTER"])
        for point in points:
            if point.cluster != -1:
                writer.writerow([point.region, point.cluster + 1])
            else:
                writer.writerow([point.region, -1])

    print(f"Saved: {output_path}")

def graph_clusters(points, month, parameters):
    output_directory = OUTPUT_FOLDER / month
    output_directory.mkdir(parents=True, exist_ok=True)

    clusters = {}

    for point in points:
        clusters.setdefault(point.cluster, []).append(point.coordinates)

    xlabel = parameters[0]

    for i in range(1, len(parameters)):
        plt.figure(figsize=(10, 7))

        for cluster_id, coordinates in clusters.items():
            x = [point[0] for point in coordinates]
            y = [point[i] for point in coordinates]

            if cluster_id == -1:
                plt.scatter(
                    x, y,
                    color="black",
                    marker="x",
                    label="Noise"
                )
            else:
                plt.scatter(
                    x, y,
                    label=f"Cluster {cluster_id + 1}"
                )

        ylabel = parameters[i]

        plt.title(f"DBSCAN Clustering - {month}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = output_directory / f"{xlabel.lower()}_vs_{ylabel.lower()}.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")

def graph_clusters_4d(points, month, parameters):
    """
    Visualize DBSCAN clustering results in four dimensions.

    X, Y, Z represent the first three parameters.
    Color represents the fourth parameter.
    """

    clusters = {}

    for point in points:
        clusters.setdefault(point.cluster, []).append(point.coordinates)

    output_directory = OUTPUT_FOLDER / month
    output_directory.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cluster_id, coordinates in clusters.items():

        x = [point[0] for point in coordinates]
        y = [point[1] for point in coordinates]
        z = [point[2] for point in coordinates]
        color_values = [point[3] for point in coordinates]

        scatter = ax.scatter(
            x, y, z,
            c=color_values,
            cmap="viridis",
            marker="x" if cluster_id == -1 else "o",
            label="Noise" if cluster_id == -1 else f"Cluster {cluster_id + 1}"
        )

    ax.set_title(f"DBSCAN 4D Clustering - {month}")
    ax.set_xlabel(parameters[0])
    ax.set_ylabel(parameters[1])
    ax.set_zlabel(parameters[2])

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label(parameters[3])

    ax.legend()
    plt.tight_layout()

    output_path = output_directory / f"clusters_4d.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")

def identify_neighbors(points: list[Point], epsilon: float):
    for point_a in points:
        for point_b in points:
            # Skip similar points
            if point_a is point_b:
                continue

            if get_euclidian_distance(point_a.coordinates, point_b.coordinates) <= epsilon:
                point_a.neighbors.append(point_b)

def cluster(point: Point, group: int):
    if point.cluster != -1:
        return

    point.cluster = group

    queue = list(point.neighbors)
    visited = set()

    while queue:
        neighbor = queue.pop()

        if neighbor in visited:
            continue

        visited.add(neighbor)

        if neighbor.cluster == -1:
            neighbor.cluster = group

        if neighbor.is_core:
            queue.extend(candidate for candidate in neighbor.neighbors if candidate not in visited and candidate.cluster == -1)

def identify_core_points(points: list[Point], minpts: int) -> list[Point]:
    core_points = []

    for point in points:
        if len(point.neighbors) + 1 < minpts:
            continue
        
        point.is_core = True
        core_points.append(point)

    return core_points


def calculate_epsilon(points_coordinates):
    # Calculate for the K Distance
    k_distance = get_k_distance(points_coordinates, MINPTS - 1)

    x = range(len(k_distance))
    knee = kneedle(x, k_distance)

    epsilon = k_distance[knee]

    return epsilon

def kneedle(x: list[float], y: list[float]):
    if len(x) != len(y):
        raise ArithmeticError("x and y must have the same length")

    if len(x) < 3:
        raise ValueError("At least three points are required.")

    x_normalized = normalize(x)
    y_normalized = normalize(y)

    differences = [
        x_normalized[i] - y_normalized[i]
        for i in range(len(x))
    ]

    return differences.index(max(differences))

def normalize(l: list[float]):
    l_max = max(l)
    l_min = min(l)
    denominator = l_max - l_min

    if denominator == 0:
        return [0.0] * len(l)

    l_normalized = []

    for value in l:
        l_normalized.append((value-l_min)/denominator)

    return l_normalized


def get_k_distance(data_points: list[float], k):
    if k < 1 or k >= len(data_points):
        raise ValueError("Invalid k value.")
    
    k_distances = []

    for point_a in data_points:
        distances = []

        for point_b in data_points:
            if point_a is point_b:
                continue

            distance = get_euclidian_distance(point_a, point_b)
            distances.append(distance)

        distances.sort()

        k_distances.append(distances[k-1])

    k_distances.sort()

    return k_distances

def graph_function(title, x, y):
    plt.plot(x, y)

    plt.title(title)

    plt.show()

def graph_local_maxima(title: str, x: list[float], y: list[float], x_local_maxima: list[float], y_local_maxima: list[float]):
    plt.plot(x, y)
    plt.scatter(x_local_maxima, y_local_maxima)

    plt.title(title)

    plt.show()

def graph_regression(title, x, y, data_points):
    plt.plot(x, y)
    plt.scatter(x, data_points)

    plt.title(title)

    plt.show()

def graph_k_distance(k_distance):
    plt.plot(range(len(k_distance)), k_distance, linestyle="-")

    plt.title("K-distance")
    plt.xlabel("No. of Paired Points")
    plt.ylabel("Distance")

    plt.show()

def get_distance_matrix(data_points):
    distance_matrix = []

    # Get the Euclidian distance for each data points
    for data_point_a in data_points:
        for data_point_b in data_points:
            # Skip similar data points
            if (data_point_a == data_point_b):
                continue

            distance = get_euclidian_distance(data_point_a, data_point_b)
            distance_matrix.append(distance)

    return distance_matrix

def get_euclidian_distance(coordinates_a, coordinates_b):
    if (len(coordinates_a) != len(coordinates_b)):
        raise ArithmeticError(f"{coordinates_a} and {coordinates_b} must have the same length")
    
    distance = 0

    for i in range(len(coordinates_a)):
        distance += (coordinates_a[i] - coordinates_b[i]) ** 2

    distance **= 0.5

    return distance

if __name__ == "__main__":
    main()
