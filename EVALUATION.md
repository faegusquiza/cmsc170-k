# Davies-Bouldin evaluation and the shared method output format

## What to tell your groupmates

> Run `python read_data.py` once to prepare the shared `processed_data` folder.
> Every method reads those same values using `read_data()` and saves
> only its own `clusters.csv` with columns `REGION,CLUSTER`. Include all 18 regions,
> including DBSCAN noise labelled `-1`. Both DBI and Silhouette combine that method's
> labels with the month's shared `standardized_values.csv`. Do not export copies
> of standardized values or regional climate averages inside method folders.

Milestone 2, page 2, defines 12 separate monthly datasets, each with 18 region
rows and four standardized features averaged over 2001-2025. Pages 5-7 and 12
specify DBI, lower-is-better interpretation, and consideration of DBSCAN noise.
The current Milestone 3 scope is method implementation, metric implementation,
and comparison tables/graphs using manually chosen settings. The PDF is the project plan;
references [1]-[3] below supply the metric formula and input conventions.

## 1. Shared data and method outputs

The shared climate data is stored once per month. Labels stay with each method:

```text
processed_data/
    metadata.json
    JAN/
        standardized_values.csv
        climate_averages.csv
    FEB/
        ...

method_results/<method>/monthly/JAN/
    clusters.csv
    ... method-specific summaries, graphs, centroids or merges
```

Repeat for FEB through DEC. Rerunning a method replaces its saved outputs.
`metadata.json` records the shared year range and region count. The original-unit
`climate_averages.csv` is used for readable plots and climate interpretation;
`standardized_values.csv` is used for clustering and evaluation. K-means'
`cluster_averages.csv` is different: it describes its selected clusters and stays
in the K-means output folder.

`clusters.csv`, showing just the first few example rows (labels are illustrative):

```csv
REGION,CLUSTER
BARMM,1
CAR,2
MIMAROPA,-1
```

Use integers for cluster labels. They may start at 0 or 1; they do not need to
be consecutive. Reserve `-1` for DBSCAN noise. Do not drop noise rows. GMM should
export the component with the largest probability for each region as its label;
membership probabilities alone are not the input to these hard-partition metrics.

`standardized_values.csv` has this January header:

```csv
REGION,MERRA2_PRECTOTCORR_JAN,MERRA2_T2M_JAN,MERRA2_WS10M_JAN,SYN1DEG_ALLSKY_SFC_SW_DWN_JAN
```

Each subsequent row contains a region name followed by its four numeric values
from `monthly_values["JAN"]`. Use the actual headers returned by
`read_data()` for that month. Exclude `ANN`, names, coordinates, centroids,
and cluster IDs from the numeric feature columns. Save all 18 region names in
each file, exactly once. The metric loader joins labels and shared values by `REGION`, so a different
CSV row order will not mix up the assignments.

The reader checks structure, numeric values and matching region names. Every
method must use the shared standardized monthly input for a fair comparison.
Missing shared files produce an error; local method copies are never used as a
fallback. If raw data or the preprocessing settings change, rerun `read_data.py`,
then rerun the methods and evaluations so their labels match the shared dataset.

## 2. How a method can return or export its result

In memory, the essential inputs are a numeric list of lists and a list of labels
in the SAME region order. Neither metric needs the method's centroids or a true
climate-class label.

```python
from read_data import read_data
from kmeans_clustering import kmeans_clustering
from davies_bouldin_index import evaluate_davies_bouldin
from read_metric_inputs import write_cluster_assignments

regions, monthly_values, monthly_headers, monthly_averages = read_data()
month = "JAN"
values = monthly_values[month]  # standardized values; 18 rows x 4 columns

labels, centroids, details = kmeans_clustering(values, k=4)
result = evaluate_davies_bouldin(values, labels)
print(result)

write_cluster_assignments(
    "method_results/kmeans/monthly/JAN",
    regions, labels
)
```

Use a fixed seed and save it with the run settings when evaluating randomized
methods, as specified in Milestone 2. Also record the year range, k, linkage or
GMM settings, and DBSCAN's epsilon/minimum-points values with the run. Those are
needed to describe and reproduce the comparison, but are not numeric features.

For hierarchical clustering, get the labels with:

```python
labels, merges = hierarchical_clustering(values, k=4, linkage="average")
```

For DBSCAN's current Point objects, construct labels in the original region order:

```python
labels = []
for point in points:
    labels.append(point.cluster)  # existing internal labels: 0, 1, ...; noise -1
```

DBSCAN now constructs its clustering points from the shared standardized
`monthly_values[month]`. Its plots display those same standardized points and
their resulting labels. It exports only its cluster assignments;
the shared tables stay in `processed_data`.

## 3. How DBI is calculated

`davies_bouldin_index.py` uses explicit loops and only Python's standard library.
The five numbered steps in `_calculate_dbi()` implement the formula in [2]:

1. Group regions according to their labels.
2. Calculate each cluster's centroid: the mean of each feature column.
3. Calculate its spread `S_i`: the average Euclidean distance of its regions to
   that centroid. Use ordinary distance, not squared distance or variance.
4. For each other cluster, calculate
   `R_ij = (S_i + S_j) / distance(centroid_i, centroid_j)`.
   Keep the largest ratio for each cluster.
5. Average those largest ratios: `DBI = sum(worst_ratios) / number_of_clusters`.

Example you can check without a library:

```python
from davies_bouldin_index import davies_bouldin_score

values = [[0], [2], [10], [12]]
labels = [1, 1, 2, 2]
print(davies_bouldin_score(values, labels))  # 0.2
```

The two centroids are 1 and 11. Each cluster's spread is 1. Their centroid
distance is 10, so both worst ratios are `(1 + 1) / 10 = 0.2`. Their mean is 0.2.

### Edge cases and DBSCAN

- Remove noise from both values and labels. Do not standardize the retained
  subset again. Return the noise count and fraction alongside the score.
- Require `2 <= number_of_clusters < number_of_evaluated_regions`, matching the
  partition-validity convention used for the project's internal metrics.
- A single-point cluster has spread 0 and is allowed when that overall count
  condition is satisfied.
- Zero clusters, one cluster, all singleton clusters, or coincident centroids
  receive `status="undefined"`, a reason, and a `None` score. This becomes a blank
  score in CSV, meaning N/A. It must not be interpreted as zero or ranked best.
- Coincident-centroid handling is an explicit project choice because the formula
  has a zero denominator. It can differ from a library's degenerate-case handling.
- Malformed data raises `ValueError`; the monthly command reports `input_error`.

DBSCAN scores use retained regions and can cover different subsets from the
other methods. As Milestone 2 states, interpret DBSCAN separately with its noise
fraction. A small DBI after dropping most regions is not enough to choose it.

## 4. Run the code

Run these commands from the project folder:

```powershell
.\.venv\Scripts\python.exe -B read_data.py
.\.venv\Scripts\python.exe -B davies_bouldin_index.py
```

Prepare the data before running the clustering methods. The final command asks
which method to evaluate. For example:

```text
Enter method (kmeans, dbscan, hierarchical): hierarchical
```

The program selects `method_results/<method>/monthly` automatically. Capitalization
and surrounding spaces are ignored. Invalid choices request another answer.
There are no folder paths or command-line arguments to configure.

The program reads method labels and shared standardized values for JAN through DEC and writes
`davies_bouldin_scores.csv` directly in the selected folder. For the example above,
the result is `method_results/hierarchical/monthly/davies_bouldin_scores.csv`. It evaluates
the saved run, including whatever k and linkage produced those labels; it does
not retrain the method or assume average linkage.

The report includes `method`, `month`, `davies_bouldin`, `status`, `reason`, total
regions, evaluated regions, non-noise cluster count, noise count, noise fraction,
and the shared standardized file used for each month.
Missing inputs produce an error row and an explanation, rather than a score.

The method column contains only `kmeans`, `dbscan`, or `hierarchical`.
There is no input-folder column. `status` says whether the score was calculated
(`ok`), cannot be defined for that partition (`undefined`), or had invalid/missing
inputs (`input_error`). `reason` explains an unsuccessful evaluation and is blank
when the score is valid. `n_noise` counts regions labelled -1, and
`noise_fraction` is that count divided by the total region count.

For only the numeric score, call `davies_bouldin_score(values, labels)`. For
reporting (including N/A cases and noise coverage), call
`evaluate_davies_bouldin(values, labels)` instead.

This implements the DBI portion. Once Silhouette is ready, its author can reuse
`load_metric_inputs()` to get exactly the same values and labels. Milestone 2's
comparison requires both scores. Compare the current method outputs on the same
monthly input and interpret the clusters using the original-unit climate averages.
The evaluator scores saved results; it does not search for or optimize settings.

## References and assistance

1. Davies, D. L., and Bouldin, D. W. (1979). A Cluster Separation Measure.
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2),
   224-227. https://doi.org/10.1109/TPAMI.1979.4766909
   Original metric reference, also listed in [2].
2. Scikit-learn, *Clustering: Davies-Bouldin Index*. Formula for scatter,
   centroid separation, pair ratios and the final index:
   https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
3. Scikit-learn, *davies_bouldin_score*. Feature-matrix and label-vector input
   convention: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html

The implementation uses these as resources, without calling scikit-learn or
copying its implementation. The noise/N/A and CSV policies above are project
integration choices. Codex assisted with code, tests, and explanations; group
members should review and be able to explain their submitted implementation.
