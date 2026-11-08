# DBSCAN Climate Clustering Implementation

## 1. Overview

This implementation applies Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to monthly climate data. It groups Philippine regions according to the similarity of their climate characteristics and identifies observations that do not belong to any sufficiently dense cluster.

The implementation processes each month independently. For every month, it determines an appropriate epsilon value using a k-distance graph and a knee-detection heuristic, identifies neighboring points and core points, and performs cluster expansion.

The program also generates visualizations of the resulting clusters and exports cluster assignments into CSV files.

## 2. System Workflow

The clustering process follows these major steps:

1. **Data Retrieval** – Load regional climate data, monthly parameter information, and monthly averages.
2. **Point Initialization** – Represent each region as a point containing its climate measurements.
3. **Epsilon Selection** – Calculate the k-distance values and estimate epsilon using knee detection.
4. **Neighbor Identification** – Determine which points are within the epsilon distance of each other.
5. **Core Point Identification** – Classify points according to the minimum number of neighboring observations.
6. **Cluster Formation** – Expand clusters by traversing the neighboring points of core points.
7. **Visualization** – Generate two-dimensional and three-dimensional visualizations of the resulting clusters.
8. **Result Export** – Save regional cluster assignments into CSV files.

## 3. Data Representation

### 3.1 Point Class

The `Point` class represents an individual geographical region and its corresponding climate measurements.

Each instance maintains the following attributes:

| Attribute | Description |
|---|---|
| `region` | Name or identifier of the geographical region. |
| `coordinates` | List of climate measurements representing the region. |
| `is_core` | Boolean indicating whether the point satisfies the core point condition. |
| `neighbors` | List of other points located within its epsilon neighborhood. |
| `cluster` | Cluster identifier assigned to the point. |

The initial cluster value is `-1`, which represents an unassigned point. Points that remain unassigned after clustering are treated as noise.

### 3.2 Climate Coordinates

Each point contains numerical measurements corresponding to the climate parameters selected for the particular month.

The coordinates are used to calculate Euclidean distances and determine neighborhood relationships.

## 4. Hyperparameter Selection

### 4.1 Minimum Points (MinPts)

The implementation uses a fixed minimum number of points:

\[
MinPts = 5
\]

This determines the minimum number of observations required within a point's epsilon neighborhood for it to qualify as a core point.

The implementation excludes the point itself when constructing its neighbor list. Therefore, it adds one when checking the MinPts condition:

\[
|N_{\epsilon}(p)| + 1 \geq MinPts
\]

For the current configuration, a point must have at least four other neighboring points to qualify as a core point.

### 4.2 Epsilon Selection

The epsilon parameter determines the maximum Euclidean distance between two points for them to be considered neighbors.

Instead of specifying a fixed epsilon value, the implementation estimates it separately for each month using the k-distance method.

#### 4.2.1 K-Distance Calculation

The `get_k_distance()` function calculates the distance from every point to its k-th nearest neighbor.

For each point:

1. Calculate its Euclidean distance to every other point.
2. Sort the calculated distances in ascending order.
3. Retrieve the distance corresponding to its k-th nearest neighbor.
4. Collect the resulting distances for all points.
5. Sort the final k-distance values in ascending order.

The implementation uses:

\[
k = MinPts - 1
\]

This is consistent with the implementation's convention of excluding the point itself from its neighbor list.

#### 4.2.2 Knee Detection

The `kneedle()` function estimates the knee of the sorted k-distance curve using maximum deviation from the normalized diagonal.

First, both the x-coordinates and k-distance values are normalized:

\[
x_i' = \frac{x_i-x_{\min}}{x_{\max}-x_{\min}}
\]

\[
y_i' = \frac{y_i-y_{\min}}{y_{\max}-y_{\min}}
\]

The difference between the normalized coordinates is then calculated:

\[
D_i = x_i' - y_i'
\]

The index corresponding to the maximum difference is selected:

\[
i^* = \arg\max_i(D_i)
\]

The corresponding k-distance value becomes the selected epsilon:

\[
\epsilon = kDistance[i^*]
\]

This approach provides a heuristic for estimating the transition between relatively small and relatively large neighborhood distances.

**Implementation note:** This is a maximum-deviation knee-detection heuristic, not the complete Kneedle algorithm. It currently does not implement local extrema detection or sensitivity-based thresholding.

## 5. DBSCAN Implementation

### 5.1 Neighbor Identification

The `identify_neighbors()` function determines the epsilon neighborhood of every point.

It iterates through every pair of points and calculates their Euclidean distance.

Two points are considered neighbors when:

\[
d(p,q) \leq \epsilon
\]

The function excludes comparisons between a point and itself.

The resulting neighborhood relationships are stored in each point's `neighbors` attribute.

### 5.2 Core Point Identification

The `identify_core_points()` function determines which points satisfy the MinPts requirement.

A point is classified as a core point when its neighborhood contains at least the required number of other observations.

Points that satisfy this condition are marked with:

```python
point.is_core = True
```

The function also returns a list of all identified core points for subsequent cluster formation.

### 5.3 Cluster Formation

The `cluster()` function performs cluster expansion using a queue-based traversal.

The procedure begins with an identified core point and assigns it to a cluster.

Its neighboring points are then placed into a queue. The algorithm repeatedly retrieves neighboring points and performs the following operations:

1. Skip points that have already been visited.
2. Assign unclassified points to the current cluster.
3. If a retrieved point is a core point, add its unclassified neighbors to the queue.
4. Continue until the queue becomes empty.

This process allows clusters to expand through density-connected core points.

Border points can be assigned to a cluster without expanding the cluster further because they do not satisfy the core point condition.

### 5.4 Cluster Identification and Noise

The main clustering procedure iterates through all identified core points.

Whenever an unassigned core point is encountered, a new cluster is created and expanded using the `cluster()` function.

Points that remain unassigned after all core points have been processed are considered noise.

The implementation uses zero-based cluster identifiers internally. During CSV export and visualization, cluster identifiers are adjusted to start from one for readability. Noise retains the identifier `-1`.

## 6. Visualization

The implementation generates two types of visualizations for every month.

### 6.1 Two-Dimensional Scatter Plots

The `graph_clusters()` function generates scatter plots using the first climate parameter as the constant x-axis.

The remaining climate parameters are individually plotted against the first parameter.

For example, given four climate parameters:

- T2M
- PRECTOTCORR
- WS10M
- ALLSKY_SFC_SW_DWN

The function generates three plots:

1. T2M versus PRECTOTCORR
2. T2M versus WS10M
3. T2M versus ALLSKY_SFC_SW_DWN

Each point is plotted according to its assigned cluster.

Noise points are represented using black cross markers, while other points are grouped according to their cluster identifiers.

The resulting graphs are saved as high-resolution PNG images with a resolution of 300 DPI.

### 6.2 Three-Dimensional Scatter Plot with a Fourth Dimension

The `graph_clusters_4d()` function generates a visualization containing four climate parameters.

The first three parameters are represented using the spatial axes:

| Dimension | Representation |
|---|---|
| First parameter | X-axis |
| Second parameter | Y-axis |
| Third parameter | Z-axis |
| Fourth parameter | Color intensity |

The fourth parameter is represented using the Viridis color map. A color bar is included to indicate the corresponding measurement values.

Cluster membership is represented through the plotted markers and legend. Noise points use cross markers, while clustered points use circular markers.

This visualization provides a way to examine four-dimensional climate observations through a three-dimensional projection with color encoding.

## 7. Result Export

### 7.1 CSV Output

The `save_clusters()` function exports the clustering assignments into a CSV file for each month.

The exported file contains two columns:

| Column | Description |
|---|---|
| `REGION` | Geographical region represented by the point. |
| `CLUSTER` | Cluster identifier assigned to the region, or `-1` if classified as noise. |

Cluster identifiers in the exported file start at one, while noise retains the value `-1`.

### 7.2 Output Directory Structure

The implementation organizes its results by month.

```text
results/
└── dbscan/
    └── monthly/
        ├── JAN/
        │   ├── clusters.csv
        │   ├── t2m_vs_prectotcorr.png
        │   ├── t2m_vs_ws10m.png
        │   ├── t2m_vs_allsky_sfc_sw_dwn.png
        │   └── clusters_4d.png
        ├── FEB/
        │   ├── clusters.csv
        │   └── ...
        └── ...
```

Each monthly directory contains its corresponding cluster assignments and visualizations.

## 8. Supporting Mathematical Functions

The implementation includes several supporting functions used throughout the clustering process.

| Function | Purpose |
|---|---|
| `get_euclidian_distance()` | Calculates the Euclidean distance between two coordinate vectors. |
| `normalize()` | Normalizes numerical values into the interval [0, 1]. |
| `get_k_distance()` | Calculates and sorts the k-distance values of all points. |
| `identify_neighbors()` | Determines the epsilon neighborhood of every point. |
| `identify_core_points()` | Identifies points satisfying the MinPts requirement. |
| `cluster()` | Expands a cluster through density-connected points. |
| `calculate_epsilon()` | Determines the epsilon value using the k-distance curve and knee detection. |

## 9. Computational Considerations

The current implementation uses pairwise distance calculations to identify neighborhoods.

For \(n\) observations with \(d\) climate features, calculating all pairwise Euclidean distances has a time complexity of approximately:

\[
O(n^2d)
\]

The neighbor-identification procedure also stores neighborhood relationships for every point.

This approach is straightforward and suitable for datasets of manageable size. However, its computational and memory requirements increase as the number of observations grows.

## 10. Implementation Limitations and Considerations

The following considerations are relevant when interpreting the current implementation:

1. **Knee Detection:** The current method uses maximum deviation from a normalized diagonal rather than the complete Kneedle algorithm. The detected point should be inspected against the actual k-distance graph.
2. **Epsilon Selection:** Epsilon is calculated independently for each month. Consequently, different months may produce different neighborhood thresholds.
3. **Cluster Order:** Border points that are reachable from multiple clusters are assigned to the cluster that reaches them first.
4. **Visualization:** Two-dimensional and three-dimensional plots are projections of the original feature space. Apparent overlap in a visualization does not necessarily indicate overlap in the complete feature space.
5. **Four-Dimensional Visualization:** The fourth parameter is represented through color rather than an additional spatial axis.
6. **Noise Classification:** Points that are not assigned to a cluster after expansion are classified as noise.

## 11. Summary

The implementation provides a complete monthly DBSCAN workflow, from climate data preparation and parameter selection to density-based clustering, visualization, and result export.

By calculating epsilon separately for each month, the implementation accommodates differences in the distribution of climate measurements across the monthly datasets.

The resulting cluster assignments can be used to examine similarities in the climate profiles of Philippine regions, while the generated visualizations provide additional insight into the distribution and separation of the identified clusters.