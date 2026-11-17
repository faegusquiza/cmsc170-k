# Philippine Monthly Climate Zone Clustering: An Analysis on Clustering Methods

**CMSC 170 Project**

**Researchers:** Marc Daniel Deang, Fatima Eryn Egusquiza, Fatima Eryn Egusquiza, John Bryan Escobia, and Jude Gabriel Padilla

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/faegusquiza/cmsc170-k)
[![Dataset](https://img.shields.io/badge/Data-NASA%20POWER-blue)](https://power.larc.nasa.gov/data-access-viewer/#single-point)



## Project Structure

The repository is organized into directories for source data, processed data, clustering methods, evaluation, tests, and the project website.

```text
cmsc170-k/
│
├── gathered_data/               # source climate data
├── processed_data/              # monthly climate averages and standardized values
├── methods/                     # K-means, hierarchical, GMM, and DBSCAN
├── method_results/              # generated clustering results and visualizations
├── metrics/                     # Silhouette Score and Davies-Bouldin Index
├── utilities/                   # data retrieval, preprocessing, and metric readers
├── tests/                       # automated tests
├── website/                     # project website
├── DBSCAN.md                    # DBSCAN documentation
├── EVALUATION.md                # evaluation documentation
├── requirements.txt
└── README.md
```



## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git

### Clone the Repository

```powershell
git clone https://github.com/faegusquiza/cmsc170-k.git
cd cmsc170-k
```

### Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```powershell
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```



## Running the Complete Pipeline

The recommended way to run the research workflow is through `main.py`.
Before running it, ensure that the required coordinate CSV files are placed
inside the `gathered_data/grid_coordinates/` directory.

```powershell
python main.py
```

When prompted:

```text
Retrieve climate data again? [y/n]:
```

- Enter `y` to retrieve fresh NASA POWER data and regenerate the processed
  datasets.
- Enter `n` to use the existing data in `processed_data/`.

The pipeline then runs K-means, hierarchical clustering, Gaussian Mixture
Model (GMM), and DBSCAN for the monthly datasets, evaluates the Silhouette
coefficient and Davies-Bouldin Index where applicable, and saves the results
under:

```text
method_results/
metric_results/
```

The pipeline uses the default parameters defined in `main.py`, including a
random seed base of `50`, average linkage for hierarchical clustering, and a
GMM convergence tolerance of `0.0001`.

## Running Individual Pipeline Stages

The following commands are optional and are intended for running or
debugging individual stages of the workflow. They are not required when using
`main.py`.

To retrieve the climate data separately:

```powershell
python -m utilities.data_retriever
```

The data retriever exports regional climate data into
`gathered_data/regions/`.

To preprocess the gathered data:

```powershell
python -m utilities.data_reader
```

The data reader outputs the shared processed datasets into `processed_data/`.

To run an individual clustering method:

```powershell
python -m methods.kmeans_clustering
python -m methods.hierarchical_clustering
python -m methods.gmm
python -m methods.dbscan_clustering
```

To evaluate method outputs separately:

```powershell
python -m metrics.davies_bouldin_index
python -m metrics.silhouette_score
```


<!-- 

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Objectives](#research-objectives)
3. [Dataset and Data Sources](#dataset-and-data-sources)
4. [Data Gathering](#data-gathering)
5. [Data Preprocessing](#data-preprocessing)
6. [Clustering Algorithms](#clustering-algorithms)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Study Delimitations](#study-delimitations)
9. [Project Structure](#project-structure)
10. [Installation and Setup](#installation-and-setup)
11. [Running the Project](#running-the-project)
12. [Contributors](#contributors)
13. [References](#references)



## Project Overview

This project explores the climate similarities among Philippine locations through unsupervised machine learning.

Using historical climate data obtained from NASA's Prediction of Worldwide Energy Resources (POWER) project, the study aims to group locations according to their climate profiles, particularly their temperature, rainfall, wind, and solar characteristics.

Rather than relying on predefined geographical or administrative classifications, the project uses clustering algorithms to identify groups of locations with similar climate patterns.

Four unsupervised learning algorithms will be explored:

- K-means Clustering
- Hierarchical Clustering
- Gaussian Mixture Model (GMM)
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

The resulting clusters will be evaluated using internal clustering metrics and interpreted in terms of their underlying climate characteristics.

## Research Objectives

The project aims to:

1. Gather historical climate data for selected Philippine locations using NASA POWER.
2. Construct climate profiles using monthly and annual climate variables.
3. Identify groups of locations with similar climate characteristics using unsupervised clustering.
4. Compare the clustering results produced by different algorithms.
5. Evaluate clustering quality using the Silhouette Score and Davies-Bouldin Index.
6. Interpret the resulting climate groupings based on their temperature, rainfall, wind, and solar characteristics.





## Study Delimitations

The study is limited by the spatial coverage and resolution of the climate data obtained from NASA.

The dataset represents climate conditions over geographic grid cells rather than providing exact measurements for every individual location. Since grid cells may cover different geographic areas and may overlap multiple administrative regions, the retrieved climate parameters may not fully represent the specific local conditions of every location.

Consequently, the resulting clusters should be interpreted as general representations of climate similarities rather than precise classifications of local climate conditions.

The study is also limited to the selected climate variables and historical period of 2001–2025.



## Dataset and Data Sources

### Primary Data Source

The project uses the [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/#single-point) to retrieve historical climate data.

NASA POWER provides meteorological and solar-related datasets derived from satellite observations and assimilation models.

**Dataset configuration:**

| Configuration | Selected Setting |
|||
| Data Access Method | Single Point |
| Data Resolution | Standard Resolution |
| Temporal Level | Monthly and Annual |
| Time Extent | 2001–2025 |
| Download Format | CSV |

**API Endpoint:**

```text
https://power.larc.nasa.gov/api/temporal/monthly/point
```

### Climate Variables

The study considers four primary climate categories.

| Category | Climate Variable | NASA Parameter | Grid Layer |
|||||
| Solar | All-Sky Surface Photosynthetically Active Radiation (PAR) Total | To be finalized | Solar |
| Temperature | Temperature at 2 Meters Minimum | To be finalized | Weather |
| Rainfall | Precipitation Average | To be finalized | Weather |
| Wind | Wind Speed at 2 Meters Minimum | To be finalized | Weather |

> **Note:** The parameter codes and exact NASA POWER parameter definitions must be verified against the final API configuration. The current extraction script uses a different set of parameter codes, documented in the Data Gathering section below.



## Data Gathering

### Overview

The data-gathering pipeline retrieves historical climate data from NASA POWER for geographic grid points within selected Philippine regions.

The collected data is organized by grid layer and region before being prepared for subsequent preprocessing and clustering.

The current implementation uses two NASA POWER grid layers:

- MERRA-2
- SYN1DEG

These layers are processed separately because they provide different climate variables.

### NASA POWER API Configuration

The current Python extraction script uses the following configuration:

| Configuration | Value |
|||
| API Endpoint | NASA POWER Monthly Point API |
| Start Year | 2001 |
| End Year | 2025 |
| Temporal Resolution | Monthly |
| Community | AG |
| Response Format | CSV |

### Grid Layers and Parameters

#### MERRA-2

The MERRA-2 grid layer currently retrieves:

| Parameter Code | Description |
|||
| `T2M` | Temperature at 2 Meters |
| `PRECTOTCORR` | Corrected Precipitation |
| `WS10M` | Wind Speed at 10 Meters |

#### SYN1DEG

The SYN1DEG grid layer currently retrieves:

| Parameter Code | Description |
|||
| `ALLSKY_SFC_SW_DWN` | All-Sky Surface Shortwave Downward Irradiance |

### Input Data

Geographic grid-point coordinates are stored in separate CSV files for each grid layer.

Each coordinate CSV contains the following columns:

| Column | Description |
|||
| `region` | Philippine region assigned to the grid point |
| `latitude` | Latitude coordinate |
| `longitude` | Longitude coordinate |
| `weight` | Area-based weight assigned to the grid point |
| `area` | Land area represented by the grid point |

Example:

```csv
region,latitude,longitude,weight,area
REGION 1,18.5,120.5,0.25,123.4
REGION 1,18.5,121.25,0.75,370.2
REGION 2,21.0,122.0,0.60,245.1
REGION 2,20.5,122.0,0.40,163.4
```

Each row represents one geographic grid point. A region may contain multiple grid points depending on the coverage of the corresponding grid layer.

#### Input Directory

```text
gathered_data/
├── grid_coordinates/
│   ├── merra2_coordinates_weighted.csv
│   └── syn1_coordinates_weighted.csv
└── grid_coordinates_unweighted/
    ├── merra2_philippine_grid_coordinates.csv
    └── syn1_philippine_grid_coordinates.csv
```

### Data Extraction Workflow

The extraction process follows these steps:

**1. Load grid coordinates**

The coordinate CSV files are loaded using `csv`. Grid points are grouped according to their corresponding Philippine regions.

**2. Retrieve climate data**

For each grid point, the script sends an HTTP GET request to the NASA POWER API using the specified coordinates, climate parameters, and temporal coverage.

**3. Validate API responses**

The script checks whether each request was successful and verifies that the expected time columns do not contain NASA POWER's `-999` missing-data placeholder.

**4. Calculate regional averages**

The monthly climate values are multiplied by their grid-point weights, accumulated for each region, and divided by the total weight.

**5. Identify grid layers**

A `GRID_LAYER` column is added to distinguish the MERRA-2 and SYN1DEG records.

**6. Combine datasets**

The resulting DataFrames are vertically concatenated to produce one dataset per region.

**7. Export CSV files**

The combined regional datasets are saved as CSV files inside the `regions/` directory.

### Regional Averaging

The current implementation calculates an area-weighted average of the climate values across all grid points within a region.

$$
\bar{x}_{r,t} =
\frac{\sum_{i=1}^{n_r}w_{r,i}x_{i,t}}
{\sum_{i=1}^{n_r}w_{r,i}}
$$

Where:

- $\bar{x}_{r,t}$: Average climate value for region $r$ at time $t$.
- $n_r$: Number of grid points belonging to region $r$.
- $w_{r,i}$: Area-based weight for grid point $i$ in region $r$.
- $x_{i,t}$: Climate value at grid point $i$ for time $t$.

### Output Data

The extraction script generates one CSV file per Philippine region.

```text
regions/
├── region_1.csv
├── region_2.csv
├── car.csv
├── region_3.csv
├── ncr.csv
├── region_4-a.csv
├── mimaropa.csv
├── region_5.csv
├── region_6.csv
├── nir.csv
├── region_7.csv
├── region_8.csv
├── region_9.csv
├── region_10.csv
├── region_11.csv
├── region_12.csv
├── region_13.csv
└── barmm.csv
```

Each output file contains the climate records retrieved from both grid layers, identified through the `GRID_LAYER` column.



## Clustering Algorithms

Four unsupervised learning algorithms will be investigated.

### 1. K-means Clustering

K-means partitions observations into a predefined number of clusters based on their distances from cluster centroids.

The algorithm follows these steps:

1. Select the desired number of clusters, $k$.
2. Initialize $k$ cluster centroids.
3. Assign each observation to its nearest centroid.
4. Recalculate each centroid using the mean of the observations assigned to its cluster.
5. Repeat the assignment and centroid-update steps until convergence or another stopping criterion is reached.

**Objective:** Minimize the within-cluster sum of squared distances.

### 2. Hierarchical Clustering

Hierarchical clustering builds a hierarchy of clusters based on the similarities between observations.

The study will explore agglomerative hierarchical clustering, which follows a bottom-up approach:

1. Initially treat each observation as an individual cluster.
2. Identify the two closest clusters based on the selected distance metric and linkage criterion.
3. Merge the two clusters.
4. Repeat until all observations belong to a single hierarchy.
5. Cut the resulting dendrogram at a selected height to obtain the desired number of clusters.

Unlike K-means, hierarchical clustering does not require the number of clusters to be specified before constructing the hierarchy.

The resulting dendrogram can also help researchers explore an appropriate number of clusters.

Reference: [Hierarchical Clustering - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/)

### 3. Gaussian Mixture Model (GMM)

Gaussian Mixture Models represent observations as combinations of multiple Gaussian probability distributions.

Unlike K-means, which assigns each observation exclusively to one cluster, GMM provides probabilistic cluster membership.

The algorithm uses the Expectation-Maximization (EM) procedure:

1. Specify the number of Gaussian components.
2. Initialize the parameters of each component, including its mean, covariance, and mixture weight.
3. **Expectation step:** Estimate the probability that each observation belongs to each component.
4. **Maximization step:** Update the component parameters based on the estimated membership probabilities.
5. Repeat the expectation and maximization steps until convergence or a specified stopping criterion is reached.

GMM is useful for exploring overlapping clusters and clusters with different covariance structures.

Reference: [Gaussian Mixture Models - IBM](https://www.ibm.com/think/topics/gaussian-mixture-model)

### 4. DBSCAN

Density-Based Spatial Clustering of Applications with Noise (DBSCAN) groups observations based on their density and identifies observations that do not belong to sufficiently dense regions.

The algorithm classifies observations into three categories:

- **Core Points:** Observations with at least `min_samples` points within their `eps` neighborhood.
- **Border Points:** Observations that have fewer than `min_samples` neighbors but fall within the neighborhood of a core point.
- **Noise Points:** Observations that do not belong to any cluster.

Clusters grow by connecting core points and their density-reachable neighbors.

#### Key Parameters

| Parameter | Description |
|||
| `eps` | Maximum distance between two observations for them to be considered neighbors. |
| `min_samples` | Minimum number of observations required within an `eps` neighborhood to form a core point. |

#### DBSCAN vs. K-means

| Feature | DBSCAN | K-means |
||||
| Cluster Shape | Can identify irregularly shaped clusters | Generally favors spherical or convex clusters |
| Number of Clusters | Not required beforehand | Must be specified beforehand |
| Noise and Outliers | Can identify noise points | Sensitive to outliers |
| Main Parameters | `eps`, `min_samples` | Number of clusters (`k`) |

Reference: [DBSCAN Clustering - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/)



## Evaluation Metrics

The clustering algorithms will be evaluated using two internal clustering metrics: Silhouette Score and Davies-Bouldin Index.

These metrics assess the compactness of clusters and the separation between them.

### 1. Silhouette Score

The Silhouette Score measures how well an observation fits within its assigned cluster compared with neighboring clusters.

Its value ranges from -1 to 1.

| Score | General Interpretation |
|||
| Above 0.71 | Strong cluster structure |
| 0.51–0.70 | Reasonable cluster structure |
| 0.26–0.50 | Weak cluster structure |
| Below 0.25 | No substantial cluster structure |

Higher scores generally indicate that observations are more similar to their assigned clusters than to neighboring clusters.

Reference: [Scikit-learn: Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

### 2. Davies-Bouldin Index

The Davies-Bouldin Index evaluates clustering quality by comparing the within-cluster spread against the separation between clusters.

Lower values generally indicate more compact and better-separated clusters.

Unlike the Silhouette Score, the Davies-Bouldin Index does not have a universally applicable threshold for determining whether a clustering result is good or bad.

A value of zero represents the ideal case.

Reference: [Scikit-learn: Davies-Bouldin Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)

### Model Comparison

Both metrics will be used to compare the clustering results produced by K-means, Hierarchical Clustering, GMM, and DBSCAN.

The general evaluation criteria are:

- Higher Silhouette Score.
- Lower Davies-Bouldin Index.
- Meaningful and interpretable climate groupings.

The preferred clustering result should balance quantitative performance with the interpretability of the resulting climate profiles.

A clustering method that achieves favorable metric values but produces groups that cannot be meaningfully explained in terms of climate characteristics may not be suitable for the study.

#### Important Evaluation Considerations

The evaluation metrics must be calculated using comparable feature representations and distance metrics.

For DBSCAN, noise observations and cases where the algorithm identifies only one cluster require special handling. The Silhouette Score and Davies-Bouldin Index are not directly applicable to every possible clustering outcome.

Reference for Silhouette interpretation: [NCBI Bookshelf - Interpretation Guidelines](https://www.ncbi.nlm.nih.gov/books/NBK610805/table/ch6.tab5/?report=objectonly)



## Project Structure

The repository is organized into directories for source data, processed data, clustering methods, evaluation, tests, and the project website.

```text
cmsc170-k/
│
├── gathered_data/              # source climate data
├── processed_data/              # monthly climate averages and standardized values
├── methods/                     # K-means, hierarchical, GMM, and DBSCAN
├── method_results/              # generated clustering results and visualizations
├── metrics/                     # Silhouette Score and Davies-Bouldin Index
├── utilities/                   # data retrieval, preprocessing, and metric readers
├── tests/                       # automated tests
├── website/                     # project website
├── DBSCAN.md                    # DBSCAN documentation
├── EVALUATION.md                # evaluation documentation
├── requirements.txt
└── README.md
```



## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git

### Clone the Repository

```powershell
git clone https://github.com/faegusquiza/cmsc170-k.git
cd cmsc170-k
```

### Create a Virtual Environment

**Windows:**

```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**

```powershell
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```



## Running the Project

Ensure that the required coordinate CSV files are placed inside the `grid_coordinates/` directory.

Run the data-gathering script:

```powershell
cd gathered_data
python gathered_data.py
```

The script retrieves the climate data and exports the resulting regional CSV files into the `regions/` directory.

Additional instructions for preprocessing, clustering, and evaluation will be added as those components are implemented.



## Contributors

| Name | Role |
|||
| John Bryan Escobia | Researcher |
| Marc Daniel Deang | Researcher |
| Fatima Eryn Egusquiza | Researcher |
| Jude Gabriel Padilla | Researcher |



## References

- [NASA POWER Data Access Viewer](https://power.larc.nasa.gov/data-access-viewer/#single-point)
- [NASA POWER API Documentation](https://power.larc.nasa.gov/docs/)
- [NASA MERRA-2 Documentation](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)
- [Hierarchical Clustering - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/)
- [Gaussian Mixture Models - IBM](https://www.ibm.com/think/topics/gaussian-mixture-model)
- [DBSCAN Clustering - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/dbscan-clustering-in-ml-density-based-clustering/)
- [Scikit-learn: Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [Scikit-learn: Davies-Bouldin Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html) -->
