"""
Monthly Agglomerative Hierarchical Clustering

Runs clustering separately for JAN to DEC using the data reader.
Each region has four parameter averages for the selected month. The years
are averaged first, then each month's columns are standardized across
regions before clustering.

Note:
    AI was used for proper code documentation and code insights.
    The code is still our work.
"""

import csv
import math
import sys
from pathlib import Path

from utilities.data_reader import MONTHS, read_data

import matplotlib
matplotlib.use("Agg") # save images without opening a separate window
import matplotlib.pyplot as plt

START_YEAR = 2001
END_YEAR = 2025
OUTPUT_FOLDER = Path("method_results/hierarchical/monthly")

# calculates the Euclidean distance between two regions' lists of climate values
def euclidean_distance(first, second):
    total = 0.0
    for j in range(len(first)): # compare the same climate column from both regions
        total += (first[j] - second[j]) ** 2 # sum((first value - second value)^2)
    return total ** 0.5 # sqrt(sum of squared differences)

# calculates the distance between two clusters using the selected linkage
def cluster_distance(first, second, distances, linkage):
    values = []
    for i in first: # go through each region in the first cluster
        for j in second: # compare it with each region in the second cluster
            values.append(distances[i][j]) # get the distance already calculated for these regions

    if linkage == "single":
        return min(values) # use the nearest pair of regions
    elif linkage == "complete":
        return max(values) # use the farthest pair of regions
    else:
        return sum(values) / len(values) # sum of pair distances / number of pairs

# starts with one cluster per region, then repeatedly merges the closest two clusters
def hierarchical_clustering(climate_values, k=4, linkage="average"):
    if not climate_values or not climate_values[0]:
        print("Error: No climate values to cluster.")
        return [], []

    num_regions = len(climate_values)
    num_columns = len(climate_values[0])

    for row in climate_values:
        if len(row) != num_columns: # all regions must have the same climate columns
            print("Error: All regions must have the same number of climate values.")
            return [], []
        
        for value in row:
            try:
                valid = math.isfinite(value) # reject infinity and NaN (not a number)
            except TypeError:
                valid = False # text cannot be used to calculate distances
            if not valid:
                print("Error: Climate values must be valid numbers.")
                return [], []

    if type(k) != int:
        print("Error: The number of clusters must be an integer.")
        return [], []
    if k < 1 or k > num_regions:
        print(f"Error: Enter a number of clusters from 1 to {num_regions}.")
        return [], []
    if linkage not in ["single", "complete", "average"]:
        print("Error: Linkage must be single, complete, or average.")
        return [], []

    # calculate the distance between every pair of regions once
    distances = []
    for i in range(num_regions):
        row = []
        for j in range(num_regions):
            row.append(euclidean_distance(climate_values[i], climate_values[j]))
        distances.append(row) # distances[i][j] is the distance from region i to region j

    clusters = {}
    for i in range(num_regions):
        clusters[i] = [i] # each region starts alone: e.g. {0: [0], 1: [1], 2: [2]}

    merges, selected_clusters = [], []
    while True:
        if len(clusters) == k:
            for cluster in clusters.values():
                selected_clusters.append(sorted(cluster)) # save a separate list of the region positions

        if len(clusters) == 1:
            break # the full merge tree is complete

        cluster_ids = sorted(clusters) # use the same order every time even when distances are tied
        least_distance = float("inf") # start with a value larger than any finite distance
        first_id, second_id = -1, -1

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)): # compare each pair only once
                first = cluster_ids[i]
                second = cluster_ids[j]
                distance = cluster_distance(clusters[first], clusters[second], distances, linkage)

                if distance < least_distance: # keep the closest pair found so far
                    least_distance = distance
                    first_id, second_id = first, second

        members = clusters[first_id] + clusters[second_id] # combine the two lists of region positions
        
        # remove the two old clusters
        clusters.pop(first_id)
        clusters.pop(second_id)

        new_id = num_regions + len(merges) # new cluster IDs start after the original region IDs
        clusters[new_id] = members
        merges.append([first_id, second_id, least_distance, len(members)]) # record the merge for the tree

    # assign cluster numbers starting at 1, in order of their first region
    selected_clusters.sort()
    labels = [0] * num_regions
    for i in range(len(selected_clusters)):
        for region in selected_clusters[i]:
            labels[region] = i + 1 # shows which cluster that region belongs to

    return labels, merges # returns the cluster assigned to each region and the full list of merges

# draws the merge tree using the merges
def draw_dendrogram(regions, labels, merges, linkage, file, month):
    num_regions = len(regions)
    remaining = [num_regions + len(merges) - 1] # start from the final merged cluster
    order = []
    while remaining:
        node = remaining.pop() # take the last cluster or region added to this list
        if node < num_regions:
            order.append(node) # an original region belongs on the bottom of the tree
        else:
            merge = merges[node - num_regions] # find which two clusters formed this cluster
            remaining.append(merge[1]) # add the right side first so the left side is processed next
            remaining.append(merge[0])

    positions = {}
    names = []
    for i in range(len(order)):
        region = order[i]
        positions[region] = [i, 0.0] # horizontal position and merge height for each original region
        names.append(f"{regions[region]} (C{labels[region]})")

    fig, ax = plt.subplots(figsize=(13, 7))
    for i in range(len(merges)):
        first, second, distance, size = merges[i]
        x1, y1 = positions[first]
        x2, y2 = positions[second]
        ax.plot([x1, x1, x2, x2], [y1, distance, distance, y2], color="steelblue") # draw the joining branches
        positions[num_regions + i] = [(x1 + x2) / 2, distance] # put the new cluster between its children

    ax.set_xticks(range(num_regions), names, rotation=70, ha="right")
    ax.set_ylabel("Euclidean linkage distance (standardized climate values)")
    ax.set_title(f"{month}: Philippine Regional Climate ({START_YEAR}-{END_YEAR})\n{linkage.capitalize()} Linkage, k = {max(labels)}")
    fig.tight_layout() # make room for the region names below the tree
    fig.savefig(file, dpi=160)
    plt.close(fig)

# writes one month's clustering assignments, merge history, and summary
def write_output(month, regions, labels, merges, k, linkage):
    folder = OUTPUT_FOLDER / month # e.g. method_results/hierarchical/monthly/month
    folder.mkdir(parents=True, exist_ok=True) # create this month's folder if needed

    # clusters
    f = open(folder / "clusters.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["REGION", "CLUSTER"])
    for i in range(len(regions)):
        writer.writerow([regions[i], labels[i]])
    f.close()

    # merge history
    f = open(folder / "merges.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["FIRST", "SECOND", "DISTANCE", "SIZE"])
    for merge in merges:
        writer.writerow(merge)
    f.close()

    # summary of clusters and their members
    lines = [f"{month} ({START_YEAR}-{END_YEAR}): Agglomerative Hierarchical Clustering"]
    lines.append(f"Linkage: {linkage}; k = {k}")
    for label in range(1, k + 1):
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

# accepts single, complete, or average linkage
def get_linkage():
    while True:
        linkage = input("Enter linkage (single, complete, average): ").strip().lower()
        if linkage == "":
            linkage = "average"
        if linkage in ["single", "complete", "average"]:
            return linkage
        print("Please enter single, complete, or average.")

# reads the 12 monthly datasets, asks for settings once, then clusters each month separately
def main():
    # .\.venv\Scripts\python.exe .\hierarchical_clustering.py")

    regions, monthly_values, monthly_headers, monthly_averages = read_data(start_year=START_YEAR, end_year=END_YEAR)
    
    if not monthly_averages:
        return # the reader already printed the error

    print(f"Regions read: {len(regions)}")
    print(f"Monthly datasets: {len(MONTHS)}\nParameters per region per month: {len(monthly_headers['JAN'])}\nYears averaged: {START_YEAR}-{END_YEAR}\n")
    k = get_k(len(regions)) # use the same k for all 12 monthly runs
    linkage = get_linkage()

    for month in MONTHS: 
        climate_values = monthly_values[month]

        labels, merges = hierarchical_clustering(climate_values, k, linkage)
        if not labels:
            return # stop if clustering reported an error

        try:
            write_output(month, regions, labels, merges, k, linkage)
        except OSError:
            print(f"Error: Cannot save {month}'s results. Check whether the output folder is writable or a file is open.")
            return
        
        file = OUTPUT_FOLDER / month / "dendrogram.png"
        try:
            draw_dendrogram(regions, labels, merges, linkage, file, month)
        except OSError:
            print(f"Error: Could not update {month}'s dendrogram.")
            return

if __name__ == "__main__":
    main()
