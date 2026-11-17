"""
Silhouette Score

Implements point-by-point cohesion/separation evaluation on each month's
standardized climate values. Higher Silhouette indicates that regions sit
comfortably inside their own climate cluster and away from the others.

The Silhouette Score is calculated as follows:
    1. Group the regions by their cluster labels.
    2. For each region, find its average distance to the other regions in its
       own cluster (cohesion, a).
    3. For each region, find its average distance to the regions of every other
       cluster, and keep the smallest one (nearest-cluster separation, b).
    4. Calculate the region's silhouette: (b - a) / max(a, b).
    5. Average the silhouettes of all regions.

Scores range from -1 to 1. Values near 1 mean compact, well-separated clusters,
values near 0 mean a region lies on the border of two clusters, and negative
values suggest a region may have been placed in the wrong cluster.

Run this file, then enter kmeans, dbscan, or hierarchical when prompted.
The score table is saved as silhouette_scores.csv in that method's
method_results/<method>/monthly folder.

Formula resource:
    https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
    Original reference: P. J. Rousseeuw, "Silhouettes: A Graphical Aid to the
    Interpretation and Validation of Cluster Analysis," Journal of Computational
    and Applied Mathematics, 20, 53-65, 1987.
    https://doi.org/10.1016/0377-0427(87)90125-7

Note:
    AI was used for proper code documentation and code insights.
    The code is still our work.
"""

import csv
import math
from pathlib import Path

from utilities.metric_inputs_reader import load_metric_inputs, validate_metric_inputs
from utilities.data_reader import MONTHS, PROCESSED_FOLDER


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

def interpret_silhouette(score):
    """
    Describe a Silhouette Score using the project's benchmark ranges.

    Milestone 1 (Benchmarks for Metric Interpretation) treats these ranges as
    general guidelines rather than strict thresholds:
        above 0.70: strong structure
        0.51 to 0.70: reasonable structure
        0.26 to 0.50: weak structure
        0.25 and below: no substantial structure

    Args:
        score (float or None): Silhouette Score, or None if it is undefined.

    Returns:
        str: The structure label, or an empty string if there is no score.
    """
    if score is None:
        return ""
    if score > 0.70:
        return "strong"
    if score > 0.50:
        return "reasonable"
    if score > 0.25:
        return "weak"
    return "no substantial structure"

def _calculate_silhouette(values, labels):
    """
    Calculate the Silhouette Score for a non-noise partition.

    Points are grouped by their labels. For every region, its average distance
    to its own cluster (a) and to the nearest other cluster (b) are found, and
    its silhouette (b - a) / max(a, b) is calculated. The regions' silhouettes
    are averaged over all regions (micro average) and also averaged per cluster
    first (macro average).

    Args:
        values (list[list[float]]): Validated feature rows with noise removed.
        labels (list[int]): Cluster labels corresponding to the feature rows.

    Returns:
        tuple: Silhouette Score (micro average over regions) and macro-averaged
            Silhouette Score (average of each cluster's mean silhouette).
            Higher values indicate more compact, better-separated clusters.

    Raises:
        ValueError: If the cluster count is unsuitable or the calculated score
            is not finite.

    Note:
        This function is called after input validation and noise filtering.
        A region alone in its cluster has silhouette 0, following [Rousseeuw
        1987]. The micro average gives every region equal weight; the macro
        average gives every cluster equal weight.
    """
    # Step 1: Put the row numbers into groups using their cluster labels.
    # Example: labels [1, 1, 2, 2] give {1: [0, 1], 2: [2, 3]}
    clusters = {}
    for i in range(len(values)):
        label = labels[i]
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    k = len(clusters)
    if k < 2 or k >= len(values):
        raise ValueError("Scoring requires 2 <= non-noise clusters < non-noise regions.")

    silhouettes = []
    for i in range(len(values)):
        own_label = labels[i]
        own_members = clusters[own_label]

        # A region alone in its cluster has no cohesion to measure.
        # By definition its silhouette is 0 (it is neither good nor bad).
        if len(own_members) == 1:
            silhouettes.append(0.0)
            continue

        # Step 2: Find cohesion a: average distance to the OTHER regions
        # in the same cluster (the region itself is not counted).
        total_distance = 0.0
        for j in own_members:
            if j != i:
                total_distance += euclidean_distance(values[i], values[j])
        a = total_distance / (len(own_members) - 1)

        # Step 3: Find separation b: average distance to each OTHER cluster.
        # Keep the smallest one; that is the region's nearest neighbouring cluster.
        b = None
        for label, members in clusters.items():
            if label == own_label:
                continue
            total_distance = 0.0
            for j in members:
                total_distance += euclidean_distance(values[i], values[j])
            average = total_distance / len(members)
            if b is None or average < b:
                b = average

        # Step 4: Compare cohesion with separation.
        # If a and b are both 0 (identical points across clusters), use 0.
        largest = max(a, b)
        if largest == 0.0:
            silhouettes.append(0.0)
        else:
            silhouettes.append((b - a) / largest)

    # Step 5: Average the silhouettes. Higher is better.
    # Micro average: every region has equal weight (the standard score).
    total = 0.0
    for value in silhouettes:
        total += value
    score = total / len(silhouettes)

    # Macro average: average each cluster first, then average the clusters,
    # so a large cluster cannot outweigh a small one.
    cluster_total = 0.0
    for members in clusters.values():
        member_total = 0.0
        for i in members:
            member_total += silhouettes[i]
        cluster_total += member_total / len(members)
    macro_score = cluster_total / k

    if not math.isfinite(score) or not math.isfinite(macro_score):
        raise ValueError("Silhouette exceeded the supported numeric range.")
    return score, macro_score

def evaluate_silhouette(values, labels):
    """
    Evaluate a clustering result and record its noise coverage.

    Inputs are validated before points labelled -1 are removed from both the
    feature rows and labels. The remaining values are not standardized again.
    If the Silhouette Score cannot be calculated for the partition, its score
    is left as None and the reason is recorded.

    Args:
        values (list[list[float]]): Feature values used for clustering, with
            one row per region. The project uses four standardized features.
        labels (list[int]): One cluster label per region, in the same order
            as values. A label of -1 represents DBSCAN noise.

    Returns:
        dict: Silhouette score, macro-averaged score, interpretation, status,
            reason, total region count, evaluated region count, non-noise
            cluster count, noise count, and noise fraction.
            An undefined score is None and has status "undefined".

    Raises:
        ValueError: If the feature rows or labels contain malformed inputs.

    Note:
        Noise regions are not given a silhouette. As in the DBI evaluator,
        DBSCAN scores cover only the retained regions and must be read
        together with the noise fraction.
    """
    # Check that the climate values are valid and each region has one cluster label.
    values, labels = validate_metric_inputs(values, labels)

    # These lists will contain only regions that belong to a cluster.
    kept_values, kept_labels = [], []
    cluster_labels = []
    for i in range(len(values)):
        # DBSCAN uses -1 for noise. Skip those regions when calculating Silhouette.
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
        "silhouette": None, "silhouette_macro": None, "interpretation": "",
        "status": "ok", "reason": "",
        "n_total": len(values), "n_evaluated": len(kept_values),
        "n_clusters": len(cluster_labels),
        "n_noise": n_noise, "noise_fraction": n_noise / len(values),
    }
    try:
        # Calculate Silhouette using only the retained regions and their cluster labels.
        score, macro_score = _calculate_silhouette(kept_values, kept_labels)
        result["silhouette"] = score
        result["silhouette_macro"] = macro_score
        result["interpretation"] = interpret_silhouette(score)
    except ValueError as error:
        # If Silhouette is undefined (e.g. only one cluster), keep the score as None
        # and save the explanation instead of stopping the whole evaluation.
        result.update(status="undefined", reason=str(error))
    return result

def silhouette_score(values, labels):
    """
    Return only the Silhouette Score for a clustering result.

    This is a shorter interface for code that needs a number instead of the
    full evaluation dictionary. Input validation and noise filtering are
    handled by evaluate_silhouette().

    Args:
        values (list[list[float]]): Feature rows used for clustering.
        labels (list[int]): Corresponding cluster labels, with -1 for noise.

    Returns:
        float: Silhouette Score (micro average) for the evaluated regions.

    Raises:
        ValueError: If the input is malformed or the partition has no defined
            Silhouette Score.
    """
    result = evaluate_silhouette(values, labels)
    if result["status"] != "ok":
        raise ValueError(result["reason"])
    return result["silhouette"]

def main():
    """
    Evaluate saved clustering results for all twelve months.

    The user chooses kmeans, dbscan, or hierarchical in the terminal. The
    corresponding results folder is selected automatically. For each month,
    the method's clusters.csv is matched with shared
    standardized values from processed_data. Silhouette is calculated, and the
    result is added to a score table.
    Missing or invalid inputs are recorded as input-error rows.

    Returns:
        int: 0 if the report is saved without input errors, or 1 if the
            folder is invalid, a month's inputs are invalid, or saving fails.

    Output:
        A silhouette_scores.csv file in the chosen method's monthly folder.
        The terminal also displays each month's score and evaluation status.
    """
    while True:
        method = input("Enter method (kmeans, dbscan, hierarchical, gaussian): ").strip().lower()
        if method in ["kmeans", "dbscan", "hierarchical", "gaussian"]:
            break
        print("Please enter kmeans, dbscan, hierarchical or gaussian.")

    # The selected method determines the folder, so no path needs to be typed.
    monthly_folder = Path("method_results") / method / "monthly"
    if not monthly_folder.is_dir():
        print(f"Error: No saved results found in {monthly_folder}. Run the method first.")
        return 1

    # Save the report beside the JAN to DEC folders.
    output = monthly_folder / "silhouette_scores.csv"
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
            result = evaluate_silhouette(values, labels)
        except (OSError, ValueError) as error:
            input_errors += 1
            result = {"silhouette": None, "silhouette_macro": None, "interpretation": "",
                      "status": "input_error", "reason": str(error),
                      "n_total": None, "n_evaluated": None, "n_clusters": None,
                      "n_noise": None, "noise_fraction": None}
        row = {"method": method, "month": month}
        row.update(result)
        row["standardized_file"] = str(PROCESSED_FOLDER / month / "standardized_values.csv")
        rows.append(row)

        if result["silhouette"] is None:
            score = "N/A"
        else:
            score = f"{result['silhouette']:.6f} ({result['interpretation']})"
        print(f"{month}: Silhouette={score}; status={result['status']}")
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

    print(f"Saved {output}; {input_errors} month(s) have input errors. Higher Silhouette is better.")
    if input_errors > 0:
        return 1
    return 0

if __name__ == "__main__":
    main()
