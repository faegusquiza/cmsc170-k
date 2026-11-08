
"""
DBSCAN Clustering Implementation

This module implements Density-Based Spatial Clustering of Applications
with Noise (DBSCAN) to group Philippine regions according to their
monthly climate profiles.

The implementation follows these general steps:
    1. Load the monthly climate data.
    2. Calculate the epsilon parameter using the sorted k-distance values.
    3. Identify the neighboring points of each data point.
    4. Identify core points based on the MinPts parameter.
    5. Expand clusters by traversing density-reachable points.
    6. Visualize and export the resulting clusters.

The module also contains supporting functions for calculating Euclidean
distances, determining k-distances, estimating epsilon, and visualizing
the intermediate results used during parameter selection.

Configuration:
    MINPTS (int): Minimum number of points required for a point to be
        considered a core point. The point itself is included in this
        count.

    OUTPUT_FOLDER (Path): Directory where the clustering results,
        visualizations, and CSV files are saved.

Note:
    AI was used for proper code documentation and code insights.
    The code is still our work.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import csv

from read_data import MONTHS, read_monthly_data, standardize_data


# Directory where monthly DBSCAN results are stored.
OUTPUT_FOLDER = Path("results/dbscan/monthly")

# DBSCAN Settings
MINPTS = 5


class Point:
    """
    Represents a data point used in DBSCAN clustering.

    Each point corresponds to a Philippine region and contains its
    climate parameter values for a particular month.

    Attributes:
        region (str): Name or identifier of the region represented
            by the point.
        coordinates (list[float]): Climate parameter values that
            represent the point in the feature space.
        is_core (bool): Indicates whether the point satisfies the
            MinPts requirement for a core point.
        neighbors (list[Point]): Other points whose Euclidean
            distance from this point is less than or equal to epsilon.
        cluster (int): Identifier of the cluster assigned to the point.
            A value of -1 indicates that the point has not been assigned
            to a cluster and is initially treated as noise.
    """

    def __init__(self, region, coordinates):
        self.region = region
        self.coordinates = coordinates
        self.is_core = False
        self.neighbors = []
        self.cluster = -1

    def __str__(self):
        """
        Return a readable representation of the point.

        Coordinate values are rounded to two decimal places to make
        the output easier to inspect.

        Returns:
            str: Formatted representation containing the coordinates,
                core-point status, and assigned cluster.
        """
        coordinates = [f"{coord:.2f}" for coord in self.coordinates]

        return f"coordinates: {coordinates}, is_core: {self.is_core}, cluster: {self.cluster}"


def main():
    """
    Execute DBSCAN clustering for every month in the dataset.

    Monthly climate data is loaded and converted into Point objects.
    Each month is clustered independently using the same MinPts
    configuration and an epsilon value calculated from its k-distance
    distribution.

    After clustering, the results are visualized using two-dimensional
    and three-dimensional plots, with the fourth parameter represented
    through color. Cluster assignments are also exported to CSV files.
    """

    regions, monthly_values, monthly_parameters, monthly_averages = read_monthly_data()

    for month in MONTHS:
        print(f"Clustering {month}")

        # Represent each region as a point in the monthly feature space.
        points = [Point(regions[i], monthly_averages[month][i]) for i in range(len(regions))]

        # Estimate epsilon using the sorted k-distance distribution.
        epsilon = calculate_epsilon([point.coordinates for point in points])

        # Start at -1 because cluster identifiers are initially
        # unassigned. The first discovered cluster will have ID 0.
        clusters_count = -1

        # Identify neighboring points using the calculated epsilon.
        identify_neighbors(points, epsilon)

        # Determine which points satisfy the MinPts requirement.
        core_points = identify_core_points(points, MINPTS)

        # Expand a cluster from every unassigned core point.
        for point in core_points:
            if point.cluster != -1:
                continue

            clusters_count += 1
            cluster(point, clusters_count)

        # Generate visualizations and export the cluster assignments.
        graph_clusters(points, month, monthly_parameters[month])
        graph_clusters_4d(points, month, monthly_parameters[month])
        save_clusters(points, month)


def save_clusters(points: list[Point], month):
    """
    Export the cluster assignments of a particular month to a CSV file.

    Each row contains a region and its corresponding cluster identifier.
    Cluster identifiers are increased by one when exported so that
    clusters are numbered starting from 1. Points classified as noise
    retain the identifier -1.

    Args:
        points (list[Point]): Points containing their final cluster
            assignments.
        month (str): Month associated with the clustering results.

    Output:
        A CSV file named 'clusters.csv' containing the columns
        REGION and CLUSTER, saved inside the corresponding monthly
        output directory.
    """

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
    """
    Generate two-dimensional scatter plots of the DBSCAN clusters.

    The first climate parameter is consistently used as the x-axis.
    Each remaining parameter is plotted separately against it on the
    y-axis, producing one figure for every parameter pair.

    Noise points are displayed using black 'x' markers, while points
    belonging to clusters are displayed using their respective
    automatically assigned colors.

    Args:
        points (list[Point]): Clustered data points.
        month (str): Month associated with the clustering results.
        parameters (list[str]): Names of the climate parameters
            represented by the point coordinates.

    Output:
        A PNG image for each parameter pair, saved inside the
        corresponding monthly output directory.
    """

    output_directory = OUTPUT_FOLDER / month
    output_directory.mkdir(parents=True, exist_ok=True)

    # Group the coordinates according to their assigned cluster.
    clusters = {}

    for point in points:
        clusters.setdefault(point.cluster, []).append(point.coordinates)

    # Keep the first parameter as the constant x-axis.
    xlabel = parameters[0]

    # Generate a separate plot for every remaining parameter.
    for i in range(1, len(parameters)):
        plt.figure(figsize=(10, 7))

        for cluster_id, coordinates in clusters.items():
            x = [point[0] for point in coordinates]
            y = [point[i] for point in coordinates]

            if cluster_id == -1:
                # Display noise points separately from actual clusters.
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

        # Use the parameter names to identify the generated figure.
        output_path = output_directory / f"{xlabel.lower()}_vs_{ylabel.lower()}.png"

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")


def graph_clusters_4d(points, month, parameters):
    """
    Visualize DBSCAN clustering results using four climate parameters.

    The first three parameters determine the spatial coordinates of
    each point in a three-dimensional scatter plot. The fourth parameter
    is represented through the color of each point using the Viridis
    colormap.

    Noise points are displayed using 'x' markers, while clustered
    points use circular markers.

    Args:
        points (list[Point]): Clustered data points.
        month (str): Month associated with the clustering results.
        parameters (list[str]): Names of the four climate parameters.

    Output:
        A PNG image named 'clusters_4d.png', saved inside the
        corresponding monthly output directory.

    Note:
        This visualization uses three spatial dimensions and color
        encoding to represent the fourth parameter. It is not a
        four-dimensional spatial plot.
    """

    # Group the coordinates according to their assigned cluster.
    clusters = {}

    for point in points:
        clusters.setdefault(point.cluster, []).append(point.coordinates)

    output_directory = OUTPUT_FOLDER / month
    output_directory.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for cluster_id, coordinates in clusters.items():

        # Extract the first three parameters as spatial coordinates.
        x = [point[0] for point in coordinates]
        y = [point[1] for point in coordinates]
        z = [point[2] for point in coordinates]

        # Use the fourth parameter to determine point colors.
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

    # Display the color scale corresponding to the fourth parameter.
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label(parameters[3])

    ax.legend()
    plt.tight_layout()

    output_path = output_directory / f"clusters_4d.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


def identify_neighbors(points: list[Point], epsilon: float):
    """
    Identify the epsilon-neighborhood of every point.

    A point is considered a neighbor of another point if their
    Euclidean distance is less than or equal to epsilon. A point
    is not included in its own neighbor list.

    The resulting neighbors are stored directly in each Point object.

    Args:
        points (list[Point]): Data points to examine.
        epsilon (float): Maximum distance between two points for
            them to be considered neighbors.

    Note:
        This implementation compares every point against every other
        point, resulting in quadratic time complexity with respect
        to the number of points.
    """

    for point_a in points:
        for point_b in points:

            # Exclude the point itself from its neighbor list.
            if point_a is point_b:
                continue

            if get_euclidian_distance(point_a.coordinates, point_b.coordinates) <= epsilon:
                point_a.neighbors.append(point_b)


def cluster(point: Point, group: int):
    """
    Expand a DBSCAN cluster starting from a core point.

    The function assigns the initial point to the specified cluster
    and explores its neighbors using a queue. When a neighboring point
    is also a core point, its neighbors are added to the queue so that
    the cluster can expand through density-connected points.

    Non-core points may be assigned to a cluster as border points,
    but they do not expand the cluster further.

    Args:
        point (Point): Unassigned core point from which cluster
            expansion begins.
        group (int): Identifier assigned to the cluster.

    Note:
        Points already assigned to another cluster are not reassigned.
        This also prevents previously processed points from being
        expanded again.
    """

    if point.cluster != -1:
        return

    # Assign the starting core point to the current cluster.
    point.cluster = group

    # Begin exploring the neighbors of the starting point.
    queue = list(point.neighbors)
    visited = set()

    while queue:
        neighbor = queue.pop()

        # Avoid processing the same point more than once.
        if neighbor in visited:
            continue

        visited.add(neighbor)

        # Assign unclassified points to the current cluster.
        if neighbor.cluster == -1:
            neighbor.cluster = group

        # Only core points can expand the cluster.
        if neighbor.is_core:
            queue.extend(
                candidate
                for candidate in neighbor.neighbors
                if candidate not in visited and candidate.cluster == -1
            )


def identify_core_points(points: list[Point], minpts: int) -> list[Point]:
    """
    Identify core points based on the MinPts density requirement.

    A point is classified as a core point if the number of points
    in its epsilon-neighborhood, including the point itself, is
    greater than or equal to MinPts.

    Since the neighbor list excludes the point itself, one is added
    to its length when checking the density requirement.

    Args:
        points (list[Point]): Data points whose core-point status
            will be determined.
        minpts (int): Minimum number of points required within
            the epsilon-neighborhood.

    Returns:
        list[Point]: Points that satisfy the MinPts requirement.
    """

    core_points = []

    for point in points:

        # Include the point itself when counting its neighborhood.
        if len(point.neighbors) + 1 < minpts:
            continue

        point.is_core = True
        core_points.append(point)

    return core_points


def calculate_epsilon(points_coordinates):
    """
    Estimate the epsilon parameter using the sorted k-distance curve.

    The k-distance of each point is calculated using MinPts - 1
    neighboring points because the point itself is excluded from
    the distance calculations.

    The resulting k-distances are sorted and passed to the knee
    detection heuristic. The distance at the detected knee index
    is selected as epsilon.

    Args:
        points_coordinates (list[list[float]]): Coordinates of all
            data points.

    Returns:
        float: Estimated epsilon value.
    """

    # Calculate the distance to the (MinPts - 1)-th nearest neighbor.
    k_distance = get_k_distance(points_coordinates, MINPTS - 1)

    x = range(len(k_distance))
    knee = kneedle(x, k_distance)

    # Select the k-distance at the detected knee position.
    epsilon = k_distance[knee]

    return epsilon


def kneedle(x: list[float], y: list[float]):
    """
    Detect a knee position using maximum deviation from a diagonal.

    Both input sequences are normalized to the range [0, 1].
    The difference between the normalized x and y coordinates is
    calculated for every position.

    The index with the largest difference is selected as the knee.

    Args:
        x (list[float]): X-axis values of the curve.
        y (list[float]): Corresponding Y-axis values of the curve.

    Returns:
        int: Index of the position with the maximum deviation.

    Raises:
        ArithmeticError: If the input sequences have different lengths.
        ValueError: If fewer than three data points are provided.

    Note:
        This is a maximum-deviation-from-the-normalized-diagonal
        heuristic. It is not a complete implementation of the
        Kneedle algorithm.
    """

    if len(x) != len(y):
        raise ArithmeticError("x and y must have the same length")

    if len(x) < 3:
        raise ValueError("At least three points are required.")

    # Normalize both axes before comparing their deviations.
    x_normalized = normalize(x)
    y_normalized = normalize(y)

    differences = [
        x_normalized[i] - y_normalized[i]
        for i in range(len(x))
    ]

    # Return the position with the greatest deviation.
    return differences.index(max(differences))


def normalize(l: list[float]):
    """
    Normalize numerical values to the range [0, 1] using min-max scaling.

    Each value is transformed using the minimum and maximum values
    of the input sequence.

    Args:
        l (list[float]): Numerical values to normalize.

    Returns:
        list[float]: Normalized values. If all input values are equal,
            a list of zeros is returned.
    """

    l_max = max(l)
    l_min = min(l)
    denominator = l_max - l_min

    # Avoid division by zero when all values are identical.
    if denominator == 0:
        return [0.0] * len(l)

    l_normalized = []

    for value in l:
        l_normalized.append((value - l_min) / denominator)

    return l_normalized


def get_k_distance(data_points: list[float], k):
    """
    Calculate the k-distance for every data point.

    The k-distance of a point is defined as its distance to its
    k-th nearest neighbor. The point itself is excluded from the
    neighbor calculations.

    The resulting k-distances are sorted in ascending order to
    produce the k-distance curve used for epsilon estimation.

    Args:
        data_points (list[list[float]]): Coordinates of all data points.
        k (int): Position of the nearest neighbor whose distance
            will be calculated. The first nearest neighbor corresponds
            to k = 1.

    Returns:
        list[float]: Sorted k-distance values.

    Raises:
        ValueError: If k is less than 1 or greater than or equal
            to the number of data points.
    """

    if k < 1 or k >= len(data_points):
        raise ValueError("Invalid k value.")

    k_distances = []

    for point_a in data_points:
        distances = []

        for point_b in data_points:

            # Exclude the point itself from the distance calculations.
            if point_a is point_b:
                continue

            distance = get_euclidian_distance(point_a, point_b)
            distances.append(distance)

        # The k-th nearest neighbor is located at index k - 1.
        distances.sort()
        k_distances.append(distances[k - 1])

    # Sort all k-distances to construct the k-distance curve.
    k_distances.sort()

    return k_distances


def graph_function(title, x, y):
    """
    Plot a function using the supplied x and y values.

    This utility is intended for visualizing numerical functions
    and intermediate calculations.

    Args:
        title (str): Title displayed above the graph.
        x (list[float]): X-axis values.
        y (list[float]): Corresponding Y-axis values.
    """

    plt.plot(x, y)
    plt.title(title)
    plt.show()


def graph_local_maxima(title: str, x: list[float], y: list[float], x_local_maxima: list[float], y_local_maxima: list[float]):
    """
    Plot a function and highlight its identified local maxima.

    Args:
        title (str): Title displayed above the graph.
        x (list[float]): X-axis values of the original function.
        y (list[float]): Corresponding Y-axis values.
        x_local_maxima (list[float]): X-coordinates of the identified
            local maxima.
        y_local_maxima (list[float]): Y-coordinates of the identified
            local maxima.
    """

    plt.plot(x, y)
    plt.scatter(x_local_maxima, y_local_maxima)

    plt.title(title)
    plt.show()


def graph_regression(title, x, y, data_points):
    """
    Visualize a regression model alongside its original data points.

    The regression model is displayed as a line, while the original
    observations are displayed as individual scatter points.

    Args:
        title (str): Title displayed above the graph.
        x (list[float]): X-axis values used by the regression model.
        y (list[float]): Predicted values generated by the model.
        data_points (list[float]): Original observed values.
    """

    plt.plot(x, y)
    plt.scatter(x, data_points)

    plt.title(title)
    plt.show()


def graph_k_distance(k_distance):
    """
    Plot the sorted k-distance values used for epsilon estimation.

    The x-axis represents the position of each point in the sorted
    k-distance sequence, while the y-axis represents its distance
    to its k-th nearest neighbor.

    Args:
        k_distance (list[float]): Sorted k-distance values.
    """

    plt.plot(range(len(k_distance)), k_distance, linestyle="-")

    plt.title("K-distance")
    plt.xlabel("No. of Paired Points")
    plt.ylabel("Distance")

    plt.show()


def get_distance_matrix(data_points):
    """
    Calculate pairwise Euclidean distances between data points.

    Each data point is compared with every other data point.
    Self-comparisons are excluded, and the resulting distances
    are stored in a flattened list.

    Args:
        data_points (list[list[float]]): Coordinates of all data points.

    Returns:
        list[float]: Flattened list containing pairwise distances.
    """

    distance_matrix = []

    for data_point_a in data_points:
        for data_point_b in data_points:

            # Exclude comparisons between a point and itself.
            if data_point_a == data_point_b:
                continue

            distance = get_euclidian_distance(data_point_a, data_point_b)
            distance_matrix.append(distance)

    return distance_matrix


def get_euclidian_distance(coordinates_a, coordinates_b):
    """
    Calculate the Euclidean distance between two points.

    The distance is computed as the square root of the sum of
    squared differences between corresponding coordinates.

    Args:
        coordinates_a (list[float]): Coordinates of the first point.
        coordinates_b (list[float]): Coordinates of the second point.

    Returns:
        float: Euclidean distance between the two points.

    Raises:
        ArithmeticError: If the coordinate sequences have different
            dimensions.
    """

    if len(coordinates_a) != len(coordinates_b):
        raise ArithmeticError(
            f"{coordinates_a} and {coordinates_b} must have the same length"
        )

    distance = 0

    # Sum the squared differences between corresponding coordinates.
    for i in range(len(coordinates_a)):
        distance += (coordinates_a[i] - coordinates_b[i]) ** 2

    # Take the square root to obtain the Euclidean distance.
    distance **= 0.5

    return distance


if __name__ == "__main__":
    main()
