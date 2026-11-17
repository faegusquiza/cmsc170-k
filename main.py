"""Run all clustering methods and evaluate their results."""

import csv
import random
from pathlib import Path

from methods import dbscan_clustering, gmm, hierarchical_clustering, kmeans_clustering
from metrics.davies_bouldin_index import evaluate_davies_bouldin
from metrics.silhouette_score import evaluate_silhouette
from utilities.data_reader import MONTHS, PROCESSED_FOLDER, prepare_monthly_data, read_data
from utilities.data_retriever import main as retrieve_data
from utilities.metric_inputs_reader import write_cluster_assignments

RANDOM_SEED = 50
DEFAULT_LINKAGE = "average"
DEFAULT_GMM_TOLERANCE = 0.0001

def load_shared_data():
    """Optionally retrieve raw data, then load the shared processed tables."""
    while True:
        answer = input("Retrieve climate data again? [y/n]: ").strip().lower()
        if answer in ("y", "yes", "Y"):
            retrieve_data()
            if not prepare_monthly_data():
                raise RuntimeError("Could not prepare processed_data from gathered_data.")
            break
        elif answer in ("", "n", "no", "N"):
            break
        print("Please answer y or n.")

    regions, monthly_values, monthly_headers, monthly_averages = read_data()
    if not regions:
        raise RuntimeError(
            f"Shared data is unavailable or invalid in {PROCESSED_FOLDER}. "
            "Choose yes at the data retrieval prompt, then run main.py again."
        )
    return regions, monthly_values, monthly_headers, monthly_averages


def evaluate_month(values, labels):
    """Run both internal metrics on one month's clustering labels."""
    return {
        "silhouette": evaluate_silhouette(values, labels),
        "davies_bouldin": evaluate_davies_bouldin(values, labels),
    }

def save_assignments(method, k, month, regions, labels):
    """Keep each run's labels in a separate method-results folder."""
    run_id = f"k_{k}" if k is not None else "dbscan"
    folder = Path("method_results") / method / "monthly" / run_id / month
    write_cluster_assignments(folder, regions, labels)

def summarize_month_scores(month_results):
    """Average defined month scores while preserving N/A as None."""
    silhouette_scores = [
        result["silhouette"]["silhouette"]
        for result in month_results.values()
        if result["silhouette"]["silhouette"] is not None
    ]
    dbi_scores = [
        result["davies_bouldin"]["davies_bouldin"]
        for result in month_results.values()
        if result["davies_bouldin"]["davies_bouldin"] is not None
    ]
    return {
        "silhouette": sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else None,
        "davies_bouldin": sum(dbi_scores) / len(dbi_scores) if dbi_scores else None,
    }

def save_metric_results(method, k, month_results):
    """Save both monthly metric reports using the existing report columns."""
    run_id = f"k_{k}" if k is not None else "dbscan"
    folder = Path("metric_results") / method / "monthly" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    for metric, filename in [("silhouette", "silhouette_scores.csv"),
                             ("davies_bouldin", "davies_bouldin_scores.csv")]:
        rows = []
        for month in MONTHS:
            row = {"method": method, "month": month}
            row.update(month_results[month][metric])
            row["standardized_file"] = (PROCESSED_FOLDER / month / "standardized_values.csv").as_posix()
            rows.append(row)
        with (folder / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = summarize_month_scores(month_results)
    return {
        "method": method, "k": k,
        "mean_silhouette": summary["silhouette"],
        "mean_davies_bouldin": summary["davies_bouldin"],
        "silhouette_months": sum(result["silhouette"]["silhouette"] is not None
                                 for result in month_results.values()),
        "davies_bouldin_months": sum(result["davies_bouldin"]["davies_bouldin"] is not None
                                    for result in month_results.values()),
        "total_months": len(month_results),
        "mean_noise_fraction": sum(result["silhouette"]["noise_fraction"]
                                   for result in month_results.values()) / len(month_results),
    }


def save_summary(rows):
    """Save run means and Table 1's sum of ranks separately for each method."""
    for row in rows:
        row.update(silhouette_rank=None, dbi_rank=None, combined_rank=None)
    for method in ["kmeans", "hierarchical", "gaussian"]:
        # Compare complete runs only, using the same twelve months for both metrics.
        candidates = [row for row in rows if row["method"] == method
                      and row["silhouette_months"] == len(MONTHS)
                      and row["davies_bouldin_months"] == len(MONTHS)]
        for row in candidates:
            # Equal scores share a rank; no extra tie-breaking rule is imposed.
            row["silhouette_rank"] = 1 + sum(other["mean_silhouette"] > row["mean_silhouette"]
                                             for other in candidates)
            row["dbi_rank"] = 1 + sum(other["mean_davies_bouldin"] < row["mean_davies_bouldin"]
                                     for other in candidates)
            row["combined_rank"] = row["silhouette_rank"] + row["dbi_rank"]
    output = Path("metric_results") / "summary.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}. Higher Silhouette and lower DBI are better.")


def run_kmeans(regions, monthly_values):
    """Run K-means for each k and evaluate every month's labels."""
    summaries = []
    for k in range(1, len(regions) + 1):
        month_results = {}
        for month in MONTHS:
            random.seed(RANDOM_SEED + k)
            labels, _, _ = kmeans_clustering.kmeans_clustering(monthly_values[month], k)
            if not labels:
                raise RuntimeError(f"K-means failed for k={k}, month={month}.")
            save_assignments("kmeans", k, month, regions, labels)
            month_results[month] = evaluate_month(monthly_values[month], labels)
        summary = summarize_month_scores(month_results)
        print_summary("K-means", k, summary)
        summaries.append(save_metric_results("kmeans", k, month_results))
    return summaries

def run_hierarchical(regions, monthly_values):
    """Run agglomerative clustering for each k and evaluate every month."""
    summaries = []
    for k in range(1, len(regions) + 1):
        month_results = {}
        for month in MONTHS:
            labels, _ = hierarchical_clustering.hierarchical_clustering(
                monthly_values[month], k=k, linkage=DEFAULT_LINKAGE
            )
            if not labels:
                raise RuntimeError(f"Hierarchical clustering failed for k={k}, month={month}.")
            save_assignments("hierarchical", k, month, regions, labels)
            month_results[month] = evaluate_month(monthly_values[month], labels)
        summary = summarize_month_scores(month_results)
        print_summary(f"Hierarchical ({DEFAULT_LINKAGE} linkage)", k, summary)
        summaries.append(save_metric_results("hierarchical", k, month_results))
    return summaries

def run_gmm(regions, monthly_values):
    """Run GMM for each component count and evaluate every month's labels."""
    summaries = []
    for k in range(1, len(regions) + 1):
        month_results = {}
        for month in MONTHS:
            random.seed(RANDOM_SEED + k)
            # The existing GMM method expects a NumPy array as its input.
            # Use the NumPy module it already imports; main.py does not import it.
            values = gmm.np.array(monthly_values[month])
            labels, _, _, _, _ = gmm.gaussian_clusters(
                values, distros=k, dims=len(values[0]), err_tolerance=DEFAULT_GMM_TOLERANCE
            )
            if len(labels) != len(regions):
                raise RuntimeError(f"GMM failed for k={k}, month={month}.")
            save_assignments("gaussian", k, month, regions, labels)
            month_results[month] = evaluate_month(monthly_values[month], labels)
        summary = summarize_month_scores(month_results)
        print_summary("GMM", k, summary)
        summaries.append(save_metric_results("gaussian", k, month_results))
    return summaries

def run_dbscan(regions, monthly_values):
    """Run the existing DBSCAN procedure once per month (no cluster k)."""
    month_results = {}
    for month in MONTHS:
        values = monthly_values[month]
        points = [dbscan_clustering.Point(regions[i], values[i]) for i in range(len(regions))]
        epsilon = dbscan_clustering.calculate_epsilon([point.coordinates for point in points])
        dbscan_clustering.identify_neighbors(points, epsilon)
        core_points = dbscan_clustering.identify_core_points(points, dbscan_clustering.MINPTS)

        cluster_id = -1
        for point in core_points:
            if point.cluster != -1:
                continue
            cluster_id += 1
            dbscan_clustering.cluster(point, cluster_id)

        labels = [point.cluster for point in points]
        save_assignments("dbscan", None, month, regions, labels)
        month_results[month] = evaluate_month(values, labels)
        silhouette = month_results[month]["silhouette"]["silhouette"]
        dbi = month_results[month]["davies_bouldin"]["davies_bouldin"]
        if silhouette is None:
            silhouette_text = "N/A"
        else:
            silhouette_text = f"{silhouette:.6f}"
        if dbi is None:
            dbi_text = "N/A"
        else:
            dbi_text = f"{dbi:.6f}"
        print(f"DBSCAN {month}: Silhouette={silhouette_text}; DBI={dbi_text}")

    summary = summarize_month_scores(month_results)
    print_summary("DBSCAN (MinPts/epsilon selected by existing procedure)", None, summary)
    return [save_metric_results("dbscan", None, month_results)]

def print_summary(method, k, summary):
    k_text = "not applicable" if k is None else str(k)
    if summary["silhouette"] is None:
        silhouette_text = "N/A"
    else:
        silhouette_text = f"{summary['silhouette']:.6f}"
    if summary["davies_bouldin"] is None:
        dbi_text = "N/A"
    else:
        dbi_text = f"{summary['davies_bouldin']:.6f}"
    print(
        f"{method}, k={k_text}: "
        f"mean Silhouette={silhouette_text}; "
        f"mean DBI={dbi_text}"
    )

def main():
    """Run the complete evaluation from the terminal."""
    try:
        regions, monthly_values, _, _ = load_shared_data()
        print(f"Loaded {len(regions)} regions across {len(MONTHS)} monthly datasets.")
        print(f"Hierarchical linkage: {DEFAULT_LINKAGE}")
        print(f"GMM convergence tolerance: {DEFAULT_GMM_TOLERANCE}")

        print("\nRunning K-means for k=1 through the number of regions...")
        summaries = run_kmeans(regions, monthly_values)
        print("\nRunning hierarchical clustering for k=1 through the number of regions...")
        summaries.extend(run_hierarchical(regions, monthly_values))
        print("\nRunning GMM for k=1 through the number of regions...")
        summaries.extend(run_gmm(regions, monthly_values))
        print("\nRunning DBSCAN once per month...")
        summaries.extend(run_dbscan(regions, monthly_values))
        save_summary(summaries)
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Clustering sweep stopped: {error}") from error

    print("\nAll clustering runs and metric evaluations are complete.")

if __name__ == "__main__":
    main()
