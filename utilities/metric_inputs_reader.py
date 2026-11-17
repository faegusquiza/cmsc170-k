"""
Evaluation Metric Input Processing

Reads, checks, and saves the monthly inputs used by the evaluation metrics.
Each method supplies clusters.csv with region labels. Standardized feature
values are read from the shared processed_data folder. Region names connect
the assignments to the shared values.
"""

import csv
import math
from numbers import Integral, Real
from pathlib import Path

from utilities.data_reader import PROCESSED_FOLDER, read_processed_file


def validate_metric_inputs(values, labels):
    """
    Validate the feature rows and their corresponding cluster labels.

    Every region must have the same number of finite numeric features and
    exactly one integer cluster label. A label of -1 represents noise.
    Valid inputs are copied into ordinary lists of floats and integers.

    Args:
        values (list[list[float]]): Feature rows used for clustering.
        labels (list[int]): Cluster label for each feature row.

    Returns:
        tuple: Validated feature rows and cluster labels as separate lists.

    Raises:
        ValueError: If inputs are empty, row sizes differ, the number of labels
            does not match the rows, or any value or label is invalid.
    """
    values = list(values)
    labels = list(labels)
    if len(values) == 0 or len(values[0]) == 0:
        raise ValueError("Feature values must contain at least one row and column.")
    if len(values) != len(labels):
        raise ValueError("Exactly one cluster label is required for each feature row.")

    num_columns = len(values[0])
    checked_values = []
    for row in values:
        if len(row) != num_columns:
            raise ValueError("Every feature row must have the same number of columns.")

        checked_row = []
        for value in row:
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError("Feature values must be finite numbers.")
            if not math.isfinite(value):
                raise ValueError("Feature values must be finite numbers.")
            checked_row.append(float(value))
        checked_values.append(checked_row)

    checked_labels = []
    for label in labels:
        if not isinstance(label, Integral) or isinstance(label, bool):
            raise ValueError("Cluster labels must be integers, with -1 reserved for noise.")
        if label < -1:
            raise ValueError("Cluster labels must be nonnegative integers or -1 for noise.")
        checked_labels.append(int(label))

    return checked_values, checked_labels


def _validate_regions(regions):
    """
    Check that every region name is nonempty and appears once.

    Args:
        regions (list[str]): Region names from a metric input file.

    Raises:
        ValueError: If a region name is empty, is not text, or is repeated.
    """
    seen_regions = []
    for region in regions:
        if not isinstance(region, str) or region.strip() == "":
            raise ValueError("Every region must have a nonempty name.")
        if region in seen_regions:
            raise ValueError("Each region must appear exactly once.")
        seen_regions.append(region)


def _read_csv(path):
    """
    Read a CSV file and check its header and row lengths.

    The file must contain a header and at least one data row. Column names
    must be unique and nonempty, and every row must match the header size.

    Args:
        path (Path): CSV file to read.

    Returns:
        tuple: Header names and data rows as separate lists.

    Raises:
        OSError: If the file cannot be opened.
        ValueError: If the file is empty or its structure is invalid.
    """
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        for row in reader:
            rows.append(row)

    if len(rows) < 2:
        raise ValueError(f"{path}: expected a header and data rows.")
    header = rows[0]
    data_rows = rows[1:]

    seen_headers = []
    for name in header:
        if name.strip() == "" or name in seen_headers:
            raise ValueError(f"{path}: column names must be nonempty and unique.")
        seen_headers.append(name)

    for row in data_rows:
        if len(row) != len(header):
            raise ValueError(f"{path}: every row must match the header's column count.")
    return header, data_rows


def load_metric_inputs(folder, data_folder=PROCESSED_FOLDER):
    """
    Read one month's feature values and cluster assignments.

    The two files are matched using REGION so that shuffled CSV rows do not
    assign a region another region's label. Returned rows follow the order
    in the shared standardized_values.csv. Noise rows are kept for the metric
    to handle. A local feature copy in the method folder is never used.

    Args:
        folder (str or Path): Method's month folder containing clusters.csv.
        data_folder (str or Path): Shared folder containing JAN to DEC folders.
            Defaults to processed_data. The month is taken from folder's name.

    Returns:
        tuple: Region names, numeric feature rows, integer cluster labels,
            and feature column names.

    Raises:
        OSError: If an input file cannot be opened.
        ValueError: If file headers, region names, values, or labels are invalid.

    Note:
        Methods and metrics use the same shared standardized inputs. This reader
        does not standardize the data again or recreate missing shared files.
    """
    folder = Path(folder)
    cluster_header, cluster_rows = _read_csv(folder / "clusters.csv")
    values_file = Path(data_folder) / folder.name / "standardized_values.csv"
    regions, values, headers = read_processed_file(values_file)
    if cluster_header != ["REGION", "CLUSTER"]:
        raise ValueError("clusters.csv must have exactly REGION,CLUSTER as its header.")
    cluster_regions = []
    for row in cluster_rows:
        cluster_regions.append(row[0])
    _validate_regions(cluster_regions)
    _validate_regions(regions)

    if len(cluster_regions) != len(regions):
        raise ValueError("The two files must contain exactly the same set of regions, including noise.")
    for region in regions:
        if region not in cluster_regions:
            raise ValueError("The two files must contain exactly the same set of regions, including noise.")

    # Look up each label using the region's name, rather than its row position.
    labels_by_region = {}
    for row in cluster_rows:
        try:
            label = int(row[1])
        except ValueError:
            raise ValueError("CLUSTER must contain integers.")
        labels_by_region[row[0]] = label

    labels = []
    for region in regions:
        labels.append(labels_by_region[region])

    values, labels = validate_metric_inputs(values, labels)
    return regions, values, labels, headers


def write_cluster_assignments(folder, regions, labels):
    """
    Save one month's method-specific cluster assignments.

    All regions, including noise, are saved. Shared climate tables are written
    only by read_data.py, so this function creates no duplicate feature files.

    Args:
        folder (str or Path): Destination for one method's month folder.
        regions (list[str]): Region names in the same order as the labels.
        labels (list[int]): Corresponding cluster labels, with -1 for noise.

    Output:
        clusters.csv containing REGION and CLUSTER in the method's month folder.

    Raises:
        OSError: If the folder cannot be created or a file cannot be written.
        ValueError: If the names or labels are invalid or their counts differ.
    """
    regions = list(regions)
    labels = list(labels)
    _validate_regions(regions)
    if len(regions) == 0 or len(regions) != len(labels):
        raise ValueError("Provide one cluster label for every region.")
    for label in labels:
        if not isinstance(label, Integral) or isinstance(label, bool) or label < -1:
            raise ValueError("Cluster labels must be nonnegative integers or -1 for noise.")

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "clusters.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["REGION", "CLUSTER"])
        for i in range(len(regions)):
            writer.writerow([regions[i], int(labels[i])])
