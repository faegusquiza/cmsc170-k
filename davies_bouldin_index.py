"""
Davies-Bouldin Index

Implements centroid/scatter evaluation on each month's standardized climate
values. Lower DBI indicates better compactness/separation.

The DBI is calculated as follows:
    1. Calculate each cluster's centroid using its four standardized climate variables.
    2. Calculate each cluster's average distance to its centroid (spread).
    3. Compare clusters using (spread A + spread B) / centroid distance.
    4. Keep each cluster's largest (worst) ratio.
    5. Average those largest ratios.

Run this file, then enter kmeans, dbscan, or hierarchical when prompted.
The score table is saved as davies_bouldin_scores.csv in that method's
results/<method>/monthly folder.

Formula resource:
    https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
    Original reference: D. L. Davies and D. W.
    Bouldin, "A Cluster Separation Measure," IEEE TPAMI, PAMI-1(2), 224-227,
    1979. https://doi.org/10.1109/TPAMI.1979.4766909

Note:
    AI was used for proper code documentation and code insights.
    The code is still our work.
"""

import csv
import math
from pathlib import Path

from read_metric_inputs import load_metric_inputs, validate_metric_inputs
from read_data import MONTHS, PROCESSED_FOLDER


def euclidean_distance(first, second):
    """
    Calculate the Euclidean distance between two feature rows.

    Corresponding feature differences are squared and added together.
    The square root of that total gives the distance between the rows.
    Both rows must have the same number of features.

    Args:
        first (list[float]): Feature values of the first point.
        second (list[float]): Feature values of the second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    total = 0.0
    for j in range(len(first)):
        total += (first[j] - second[j]) ** 2
    return math.sqrt(total)

def _calculate_dbi(values, labels):
    """
    Calculate the Davies-Bouldin Index for a non-noise partition.

    Points are grouped by their labels. Each cluster's centroid and average
    distance to that centroid are calculated. Every pair of clusters is then
    compared using combined spread divided by centroid distance. The largest
    ratio for each cluster is kept, and those ratios are averaged.

    Args:
        values (list[list[float]]): Validated feature rows with noise removed.
        labels (list[int]): Cluster labels corresponding to the feature rows.

    Returns:
        float: Davies-Bouldin Index. Lower values indicate more compact,
            better-separated clusters.

    Raises:
        ValueError: If the cluster count is unsuitable, two centroids coincide,
            or the calculated score is not finite.

    Note:
        This function is called after input validation and noise filtering.
        Every cluster has equal weight in the final average.
    """
    # Step 1: Put the feature rows into groups using their cluster labels.
    # Example: {1: [[0], [2]], 2: [[10], [12]]}
    clusters = {}
    for i in range(len(values)):
        label = labels[i]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(values[i])

    k = len(clusters)
    if k < 2 or k >= len(values):
        raise ValueError("Scoring requires 2 <= non-noise clusters < non-noise regions.")

    centroids, scatters = [], []
    for members in clusters.values():
        # Step 2: Find this cluster's centroid (the mean of each column).
        centroid = []
        for j in range(len(values[0])):
            total = 0.0
            for row in members:
                total += row[j]
            centroid.append(total / len(members))
        centroids.append(centroid)

        # Step 3: Find its spread: average distance to the centroid.
        # A single-point cluster has spread 0.
        total_distance = 0.0
        for row in members:
            total_distance += euclidean_distance(row, centroid)
        scatter = total_distance / len(members)
        scatters.append(scatter)

    # Step 4: Compare each cluster with every OTHER cluster.
    # Keep its largest (worst) ratio of spread to separation.
    worst_ratios = []
    for i in range(k):
        worst = 0.0
        for j in range(k):
            if i == j:
                continue
            separation = euclidean_distance(centroids[i], centroids[j])
            if separation == 0.0:
                # Explicit project policy: report N/A rather than an artificial 0.
                raise ValueError("Two cluster centroids coincide; their DBI ratio has zero denominator.")
            ratio = (scatters[i] + scatters[j]) / separation
            if ratio > worst:
                worst = ratio
        worst_ratios.append(worst)

    # Step 5: Average the worst ratios. Lower DBI is better.
    # Every cluster has equal weight in this final average.
    total_ratio = 0.0
    for ratio in worst_ratios:
        total_ratio += ratio
    score = total_ratio / k
    if not math.isfinite(score):
        raise ValueError("DBI exceeded the supported numeric range.")
    return score

def evaluate_davies_bouldin(values, labels):
    """
    Evaluate a clustering result and record its noise coverage.

    Inputs are validated before points labelled -1 are removed from both the
    feature rows and labels. The remaining values are not standardized again.
    If DBI cannot be calculated for the partition, its score is left as None
    and the reason is recorded.

    Args:
        values (list[list[float]]): Feature values used for clustering, with
            one row per region. The project uses four standardized features.
        labels (list[int]): One cluster label per region, in the same order
            as values. A label of -1 represents DBSCAN noise.

    Returns:
        dict: DBI score, status, reason, total region count, evaluated region
            count, non-noise cluster count, noise count, and noise fraction.
            An undefined score is None and has status "undefined".

    Raises:
        ValueError: If the feature rows or labels contain malformed inputs.

    Note:
        A single-point cluster has spread 0. It is allowed when the overall
        cluster count satisfies the evaluator's scoring requirements.
    """
    # Check that the climate values are valid and each region has one cluster label.
    values, labels = validate_metric_inputs(values, labels)

    # These lists will contain only regions that belong to a cluster.
    kept_values, kept_labels = [], []
    cluster_labels = []
    for i in range(len(values)):
        # DBSCAN uses -1 for noise. Skip those regions when calculating DBI.
        if labels[i] != -1:
            # Keep the climate row and its matching label together, in the same order.
            kept_values.append(values[i])
            kept_labels.append(labels[i])
            # Record each cluster label once so we can count the actual clusters.
            # For example, labels [1, 1, 2, 2] give cluster_labels [1, 2].
            if labels[i] not in cluster_labels:
                cluster_labels.append(labels[i])

    # Regions removed from the original list are the noise points.
    n_noise = len(values) - len(kept_values)

    # Prepare the report. None means there is no score yet; it does not mean zero.
    # noise_fraction is the share removed as noise, e.g. 2 out of 10 gives 0.2.
    result = {
        "davies_bouldin": None,
        "status": "ok", "reason": "",
        "n_total": len(values), "n_evaluated": len(kept_values),
        "n_clusters": len(cluster_labels),
        "n_noise": n_noise, "noise_fraction": n_noise / len(values),
    }
    try:
        # Calculate DBI using only the retained regions and their cluster labels.
        result["davies_bouldin"] = _calculate_dbi(kept_values, kept_labels)
    except ValueError as error:
        # If DBI is undefined (e.g. only one cluster), keep the score as None
        # and save the explanation instead of stopping the whole evaluation.
        result.update(status="undefined", reason=str(error))
    return result

def davies_bouldin_score(values, labels):
    """
    Return only the Davies-Bouldin score for a clustering result.

    This is a shorter interface for code that needs a number instead of the
    full evaluation dictionary. Input validation and noise filtering are
    handled by evaluate_davies_bouldin().

    Args:
        values (list[list[float]]): Feature rows used for clustering.
        labels (list[int]): Corresponding cluster labels, with -1 for noise.

    Returns:
        float: Davies-Bouldin Index for the evaluated regions.

    Raises:
        ValueError: If the input is malformed or the partition has no defined
            DBI score.
    """
    result = evaluate_davies_bouldin(values, labels)
    if result["status"] != "ok":
        raise ValueError(result["reason"])
    return result["davies_bouldin"]

def main():
    """
    Evaluate saved clustering results for all twelve months.

    The user chooses kmeans, dbscan, or hierarchical in the terminal. The
    corresponding results folder is selected automatically. For each month,
    the method's clusters.csv is matched with shared
    standardized values from processed_data. DBI is calculated, and the result
    is added to a score table.
    Missing or invalid inputs are recorded as input-error rows.

    Returns:
        int: 0 if the report is saved without input errors, or 1 if the
            folder is invalid, a month's inputs are invalid, or saving fails.

    Output:
        A davies_bouldin_scores.csv file in the chosen method's monthly folder.
        The terminal also displays each month's score and evaluation status.
    """
    while True:
        method = input("Enter method (kmeans, dbscan, hierarchical, gaussian): ").strip().lower()
        if method in ["kmeans", "dbscan", "hierarchical", "gaussian"]:
            break
        print("Please enter kmeans, dbscan, or hierarchical, gaussian.")

    # The selected method determines the folder, so no path needs to be typed.
    monthly_folder = Path("results") / method / "monthly"
    if not monthly_folder.is_dir():
        print(f"Error: No saved results found in {monthly_folder}. Run the method first.")
        return 1

    # Save the report beside the JAN to DEC folders.
    output = monthly_folder / "davies_bouldin_scores.csv"
    rows = []
    input_errors = 0
    for month in MONTHS:
        folder = monthly_folder / month
        try:
            regions, values, labels, headers = load_metric_inputs(folder, PROCESSED_FOLDER)
            # This project uses four standardized climate features per month.
            if len(headers) != 4:
                raise ValueError(f"Expected four feature headers ending in _{month}.")
            for header in headers:
                if not header.endswith("_" + month):
                    raise ValueError(f"Expected four feature headers ending in _{month}.")
            result = evaluate_davies_bouldin(values, labels)
        except (OSError, ValueError) as error:
            input_errors += 1
            result = {"davies_bouldin": None, "status": "input_error", "reason": str(error),
                      "n_total": None, "n_evaluated": None, "n_clusters": None,
                      "n_noise": None, "noise_fraction": None}
        row = {"method": method, "month": month}
        row.update(result)
        row["standardized_file"] = str(PROCESSED_FOLDER / month / "standardized_values.csv")
        rows.append(row)

        if result["davies_bouldin"] is None:
            score = "N/A"
        else:
            score = f"{result['davies_bouldin']:.6f}"
        print(f"{month}: DBI={score}; status={result['status']}")
        if result["reason"] != "":
            print("Reason:", result["reason"])

    try:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    except OSError:
        print("Error: Could not save the score table. Check whether the file is open or the folder is writable.")
        return 1

    print(f"Saved {output}; {input_errors} month(s) have input errors. Lower DBI is better.")
    if input_errors > 0:
        return 1
    return 0

if __name__ == "__main__":
    main()
