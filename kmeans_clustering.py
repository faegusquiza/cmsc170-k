"""
Monthly K-means Clustering

Runs clustering separately for JAN to DEC using the data reader.
Each region has four parameter averages for the selected month. The years
are averaged first, then each month's columns are standardized across
regions before clustering.

The initial centroids are k randomly picked regions. Each region is then
assigned to its nearest centroid, and each centroid is moved to the average
of its regions, until the centroids no longer change. Because the starting
regions are random, the clusters can differ from one run to the next.

Note:
    AI was used for proper code documentation and code insights.
    The code is still our work.
"""

import csv
import math
import random
from pathlib import Path

from read_data import MONTHS, read_monthly_data, standardize_data

import matplotlib
matplotlib.use("Agg") # save images without opening a separate window
import matplotlib.pyplot as plt

START_YEAR = 2001
END_YEAR = 2025
OUTPUT_FOLDER = Path("results/kmeans/monthly")


# calculates the Euclidean distance between two regions' lists of climate values
def euclidean_distance(first, second):
    total = 0.0
    for j in range(len(first)): # compare the same climate column from both regions
        total += (first[j] - second[j]) ** 2 # sum((first value - second value)^2)
    return total ** 0.5 # sqrt(sum of squared differences)

# assigns every region to its nearest centroid
def assign_clusters(climate_values, centroids):
    assignments = []
    for i in range(len(climate_values)):
        least_distance = float("inf") # start with a value larger than any finite distance
        nearest = -1

        for j in range(len(centroids)):
            distance = euclidean_distance(climate_values[i], centroids[j])
            if distance < least_distance: # keep the closest centroid found so far
                least_distance = distance
                nearest = j

        assignments.append(nearest) # assignments[i] is the centroid position of region i
    return assignments

# moves each centroid to the average of the regions assigned to it
def update_centroids(climate_values, assignments, centroids):
    num_columns = len(climate_values[0])
    new_centroids = []

    for j in range(len(centroids)):
        totals = [0.0] * num_columns
        count = 0

        for i in range(len(climate_values)): # gather all the regions assigned to this centroid
            if assignments[i] == j:
                for c in range(num_columns):
                    totals[c] += climate_values[i][c]
                count += 1

        if count != 0:
            new_centroids.append([total / count for total in totals]) # sum of values / number of regions
        else:
            new_centroids.append(list(centroids[j])) # keep the centroid if no region is assigned to it

    return new_centroids

# starts with k random regions as centroids, then repeatedly regroups the regions and moves the centroids
def kmeans_clustering(climate_values, k=4):
    if not climate_values or not climate_values[0]:
        print("Error: No climate values to cluster.")
        return [], [], {}

    num_regions = len(climate_values)
    num_columns = len(climate_values[0])

    for row in climate_values:
        if len(row) != num_columns: # all regions must have the same climate columns
            print("Error: All regions must have the same number of climate values.")
            return [], [], {}

        for value in row:
            try:
                valid = math.isfinite(value) # reject infinity and NaN (not a number)
            except TypeError:
                valid = False # text cannot be used to calculate distances
            if not valid:
                print("Error: Climate values must be valid numbers.")
                return [], [], {}

    if type(k) != int:
        print("Error: The number of clusters must be an integer.")
        return [], [], {}
    if k < 1 or k > num_regions:
        print(f"Error: Enter a number of clusters from 1 to {num_regions}.")
        return [], [], {}

    # choose k different random regions as the initial centroids
    initial_indices = []
    while len(initial_indices) < k:
        random_index = random.choice(range(num_regions))
        if random_index not in initial_indices: # skip a region that is already a centroid
            initial_indices.append(random_index)

    centroids = []
    for index in initial_indices:
        centroids.append(list(climate_values[index])) # each centroid starts on one region

    assignments = assign_clusters(climate_values, centroids)
    iterations = 0

    while True:
        new_centroids = update_centroids(climate_values, assignments, centroids)
        iterations += 1

        if new_centroids == centroids:
            break # stop when the centroids no longer change

        centroids = new_centroids
        assignments = assign_clusters(climate_values, centroids) # regroup using the moved centroids

    # assign cluster numbers starting at 1, in order of their first region
    # (the same numbering used by the hierarchical clustering results)
    new_numbers = {}
    for position in assignments:
        if position not in new_numbers:
            new_numbers[position] = len(new_numbers) + 1

    labels = []
    for position in assignments:
        labels.append(new_numbers[position]) # shows which cluster that region belongs to

    final_centroids = [None] * len(new_numbers)
    for position in new_numbers:
        final_centroids[new_numbers[position] - 1] = centroids[position] # final_centroids[0] belongs to cluster 1

    details = {
        "initial_indices": initial_indices,
        "iterations": iterations,
    }

    return labels, final_centroids, details # returns each region's cluster, the final centroids, and the run details

# averages each cluster's original values so the clusters can be read in their real units
def cluster_averages(averages, labels):
    num_columns = len(averages[0])
    results = []

    for label in range(1, max(labels) + 1):
        totals = [0.0] * num_columns
        count = 0
        for i in range(len(averages)):
            if labels[i] == label:
                for c in range(num_columns):
                    totals[c] += averages[i][c]
                count += 1
        results.append([total / count for total in totals])

    return results

# shortens a column name for the plots ("MERRA2_T2M_JAN" to "T2M")
def short_name(header):
    parts = header.split("_")
    return "_".join(parts[1:-1]) # drop the grid layer and the month

# draws two scatter plots that together show all four climate values, colored by cluster
def draw_clusters(regions, averages, headers, labels, file, month):
    names = [short_name(header) for header in headers]
    pairs = [] # pick the columns to plot by name so the column order does not matter
    for x_name, y_name in [["PRECTOTCORR", "T2M"], ["WS10M", "ALLSKY_SFC_SW_DWN"]]:
        if x_name in names and y_name in names:
            pairs.append([names.index(x_name), names.index(y_name)])

    if not pairs: # fall back to the first two columns if the parameters change
        pairs.append([0, 1 if len(headers) > 1 else 0])

    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
    if len(pairs) == 1:
        axes = [axes]

    colors = plt.get_cmap("tab10")
    for p in range(len(pairs)):
        ax = axes[p]
        x_col, y_col = pairs[p]

        for label in range(1, max(labels) + 1):
            xs, ys = [], []
            for i in range(len(regions)):
                if labels[i] == label:
                    xs.append(averages[i][x_col])
                    ys.append(averages[i][y_col])
            ax.scatter(xs, ys, color=colors((label - 1) % 10), label=f"Cluster {label}", s=60)

        for i in range(len(regions)): # write each region's name beside its point
            ax.annotate(regions[i], (averages[i][x_col], averages[i][y_col]), fontsize=7, xytext=(4, 3), textcoords="offset points")

        ax.set_xlabel(headers[x_col])
        ax.set_ylabel(headers[y_col])
        ax.legend(fontsize=8)

    fig.suptitle(f"{month}: Philippine Regional Climate ({START_YEAR}-{END_YEAR})\nK-means, k = {max(labels)}")
    fig.tight_layout()
    fig.savefig(file, dpi=160)
    plt.close(fig)

# writes one month's assignments, original averages, standardized values, and centroids
def write_output(month, regions, averages, climate_values, headers, labels, centroids, details, k):
    folder = OUTPUT_FOLDER / month # e.g. results/kmeans/monthly/month
    folder.mkdir(parents=True, exist_ok=True) # create this month's folder if needed

    # clusters
    f = open(folder / "clusters.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["REGION", "CLUSTER"])
    for i in range(len(regions)):
        writer.writerow([regions[i], labels[i]])
    f.close()

    # climate averages
    f = open(folder / "climate_averages.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["REGION"] + headers)
    for i in range(len(regions)):
        writer.writerow([regions[i]] + averages[i])
    f.close()

    # standardized climate values
    f = open(folder / "standardized_values.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["REGION"] + headers)
    for i in range(len(regions)):
        writer.writerow([regions[i]] + climate_values[i])
    f.close()

    # final centroids in standardized values (what K-means used)
    f = open(folder / "centroids.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["CLUSTER"] + headers)
    for i in range(len(centroids)):
        writer.writerow([i + 1] + centroids[i])
    f.close()

    # each cluster's average in the original units (for interpreting the climate groups)
    f = open(folder / "cluster_averages.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["CLUSTER"] + headers)
    profiles = cluster_averages(averages, labels)
    for i in range(len(profiles)):
        writer.writerow([i + 1] + profiles[i])
    f.close()

    # summary of clusters and their members
    starting_regions = [regions[index] for index in details["initial_indices"]]
    lines = [f"{month} ({START_YEAR}-{END_YEAR}): K-means Clustering"]
    lines.append(f"k = {k}")
    lines.append(f"Initial centroids: {', '.join(starting_regions)}")
    lines.append(f"Iterations: {details['iterations']}")
    for label in range(1, max(labels) + 1):
        members = []
        for i in range(len(regions)):
            if labels[i] == label:
                members.append(regions[i])
        lines.append(f"Cluster {label}: {', '.join(members)}")

    summary = "\n".join(lines)
    print("\n" + summary)

    f = open(folder / "summary.txt", "w", encoding="utf-8")
    f.write(summary + "\n")
    f.close()

# asks for k if the input is not a valid number of clusters
def get_k(num_regions):
    while True:
        try:
            k = int(input(f"Enter the number of clusters (1-{num_regions}): "))
        except ValueError:
            print("Please enter a whole number.")
            continue

        if 1 <= k <= num_regions:
            return k
        print(f"Please enter a number from 1 to {num_regions}.")

# reads the 12 monthly datasets, asks for k once, then clusters each month separately
def main():
    # .\.venv\Scripts\python.exe .\kmeans_clustering.py

    regions, monthly_averages, monthly_headers = read_monthly_data(start_year=START_YEAR, end_year=END_YEAR, standardize=False)

    if not monthly_averages:
        return # the reader already printed the error

    print(f"Regions read: {len(regions)}")
    print(f"Monthly datasets: {len(MONTHS)}\nParameters per region per month: {len(monthly_headers['JAN'])}\nYears averaged: {START_YEAR}-{END_YEAR}\n")
    k = get_k(len(regions)) # use the same k for all 12 monthly runs

    for month in MONTHS:
        averages = monthly_averages[month] # 18 rows with four averages each in the current data
        headers = monthly_headers[month] # this month's four parameter names
        climate_values = standardize_data(averages) # standardize only this month's four columns across regions
        if not climate_values:
            return

        labels, centroids, details = kmeans_clustering(climate_values, k)
        if not labels:
            return # stop if clustering reported an error

        try:
            write_output(month, regions, averages, climate_values, headers, labels, centroids, details, k)
        except OSError:
            print(f"Error: Cannot save {month}'s results. Check whether the output folder is writable or a file is open.")
            return

        file = OUTPUT_FOLDER / month / "clusters.png"
        try:
            draw_clusters(regions, averages, headers, labels, file, month)
        except OSError:
            print(f"Error: Could not update {month}'s cluster plot.")
            return

if __name__ == "__main__":
    main()
