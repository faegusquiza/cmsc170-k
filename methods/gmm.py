import csv
import numpy as np
import math
import random
import csv
from utilities.data_reader import MONTHS, read_monthly_data, standardize_data
from pathlib import Path

import matplotlib
matplotlib.use("Agg") # save images without opening a separate window
import matplotlib.pyplot as plt

START_YEAR = 2001
END_YEAR = 2025
OUTPUT_FOLDER = Path("method_results/gaussian/monthly")

def gen_global_cov(data, dimensions):
    #The data variable should be in order of:
    #     dimension1  dimension2  dimension3    ...
    # pt1      67          68          69       ...
    # pt2      420         ...         ...      ...
    # pt3      ...         ...         ...      ...
    # ...      ...         ...         ...      ...
    # so that would be like... data[points][dimension]

    #Let N be the total number of datapoints
    N = len(data)

    # Lets numpify the data for better code ergonomics


    #we must calculate the mean...
    #this mean is a vector with D values (D as in dimensions)
    #calculates the mean among the D dimensions
    x_bar = [0]*dimensions
    for i in range(dimensions):
        x_bar[i] = sum(data[: , i ])/N



    #now that we have all this stuffs, we can go ahead and do the iterating.
    #summation is a DxD matrix
    summation = np.full((dimensions, dimensions), 0)
    for i in range(N):
        summation = summation + np.outer( (data[i] - x_bar) ,((data[i] -x_bar)) )

    #after that, we divide it by N since we will be dealing with a population
    #and assign it to our covar_matrix
    covar_matr = summation / N 


    # lets un-numpify the data for uniformity or something

    return covar_matr

def prob_density_func( datapt , mean, covar, dimensions):

    #Turn the data and the mean into numpy again
    datapt = np.array(datapt)
    mean = np.array(mean)
    covar = np.array(covar)

    #lets split this to numerator and denominator because there are alot to cover

    diff = datapt-mean
    numerator = math.e**((-1/2) * np.inner(diff, np.linalg.inv(covar) @ (diff) ) )

    denominator = ((2*math.pi)**(dimensions/2)) * (np.linalg.det(covar)**(1/2))

    density_probability = numerator/denominator 

    
    return density_probability

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


#Calculates the actual gaussian clusterings
#OUTPUT (in order):
# datapoint groups (numpy matrix)
# means per clusters (numpy list)
# weights per clusters(numpy list)
# covariance matrices per clusters(numpy matrix)
# details (dict{"initial_indices": #, "iterations": # })
def gaussian_clusters(data, distros, dims, err_tolerance):
    regularizer = 1e-6 * np.eye(dims)

    for row in data:
        if len(row) != dims: # all regions must have the same climate columns
            print("Error: All regions must have the same number of climate values.")
            return [], [], [], [], {}

        for value in row:
            try:
                valid = math.isfinite(value) # reject infinity and NaN (not a number)
            except TypeError:
                valid = False # text cannot be used to calculate distances
            if not valid:
                print("Error: Climate values must be valid numbers.")
                return [], [], [], [], {}


    #initialize the means such that it takes k random samples from our data
    initial_indices = []
    means = []
    while len(means) < distros:
        random_index = random.randrange(len(data))
        random_mean = data[random_index]
        if not any(np.array_equal(random_mean, x) for x in means): # skip a region that is already a centroid
            initial_indices.append(random_index)
            means.append(random_mean)
    means = np.array(means)
    
    # Probability of all detaset towards the k clusters
    # This will be a list, where the elements is a list with k elements
    # THIS IS A NUMPY ARRAY
    probabilities = np.zeros((len(data), distros))

    #initialize the weights by assuming all clusters are equally likely using uniform distribution
    weights = [1/distros]*distros
    weights = np.array(weights)

    #Initialize covariance matrices
    cov_mat = np.array([
        gen_global_cov(data=data, dimensions=dims) + regularizer
        for _ in range(distros)
    ])
    initial_covars = cov_mat
    #initiate log_likelihood to infinity
    log_like_old = math.inf

    datapt_group = [0 for _ in range(len(data))]

    
    # We now have our needed data...
    # we do the looping!
    iterations = 0
    while(1):
        iterations += 1
        # print("iterations:", iterations)
        # This will be the expectation step
        # Loop for the data
        for i in range(len(data)):
            #Loop for the data's dimensions
            for k in range(distros):
                #lets subdivide this into numerator and denominator
                numerator = weights[k] * prob_density_func(data[i], means[k], cov_mat[k], dims)

                #the denominator needs iterations
                denominator = 0 
                for l in range(distros):
                    denominator += weights[l] * prob_density_func(data[i], means[l], cov_mat[l], dims)

                probabilities[i][k] = numerator/denominator
                
            
        # Now that we have the expectation step, check where each data points belongs
        datapt_group = [[] for _ in range(len(data))]
        for j in range(len(data)):
            max = -math.inf
            max_idx = math.inf
            for l in range(distros): # I think this will make an error
                if probabilities[j][l] > max:
                    max = probabilities[j][l]
                    max_idx = l
            datapt_group[j] = max_idx + 1


        #Now, we go to the maximization step
        # Calculation for new means
        for k in range(distros):
            #ok we split them to numerators and denominators again
            #both numerator and denominator need summations
            #we can actually use this loop to get all the 2 other gauss variables
            mean_numerator = 0
            cov_numerator = 0
            mean_and_cov_denom_and_weigth_sum = 0

            #for a while, we turn the data and mean into a numpy matrix

            for i in range(len(data)):
                #for a while, we turn the data and mean into a numpy matrix
                mean_numerator = mean_numerator + (probabilities[i][k]* data[i])
                mean_and_cov_denom_and_weigth_sum = mean_and_cov_denom_and_weigth_sum + probabilities[i][k]

            means[k] = mean_numerator/mean_and_cov_denom_and_weigth_sum
            weights[k] = mean_and_cov_denom_and_weigth_sum/len(data)

            for i in range(len(data)):
                cov_numerator = cov_numerator + probabilities[i][k] * np.outer(data[i] - means[k] , data[i] - means[k])

            cov_mat[k] = cov_numerator / mean_and_cov_denom_and_weigth_sum + regularizer

        # print("new parameters:\n means:\n", means,"\n covar:\n", cov_mat, "\n weights:\n", weights )

        #After that, we calculate the log likelihood for the convergence checking part
        #this log_like_new variable will be the outer loop
        log_like_new = 0
        
        for i in range(len(data)):

            inner_log_sum = 0
            for k in range(distros):
                inner_log_sum += weights[k] * prob_density_func(data[i], means[k], cov_mat[k], dims)

            log_like_new += math.log(inner_log_sum)

        if(abs(log_like_new - log_like_old) < err_tolerance):
            # print("convergence detected")
            break
        log_like_old = log_like_new 
        # print("New means:\n", means, "\nNew Weights:\n", weights, "\nNew_covars\n", cov_mat)
    details = {
        "initial_indices" : initial_indices,
        "initial_covariances" : initial_covars ,
        "iterations": iterations
    }

    return datapt_group, means, weights, cov_mat, details

def write_output(month, regions, averages, climate_values, headers, labels, centroids, details, k):
    folder = OUTPUT_FOLDER / month # e.g. method_results/gaussian/monthly/{month}
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
        writer.writerow([regions[i]] + climate_values[i].tolist())
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
    lines = [f"{month} ({START_YEAR}-{END_YEAR}): Gaussian Mixture Model Clustering"]
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

def main():

    regions, monthly_values, monthly_headers, monthly_averages = read_monthly_data()

    print("Gaussion Mixture Model Clustring for the climate of the regions in the Philippines.")
    print(f"Regions read: {len(regions)}")
    print(f"Monthly datasets: {len(MONTHS)}\nParameters per region per month: {len(monthly_headers['JAN'])}\nYears averaged: {START_YEAR}-{END_YEAR}\n")

    distros = get_k(len(regions))
    print("Enter the convergence error tolerance (e.g. 0.00001):")
    err_tolerance = float(input())

    for month in MONTHS:
        averages = monthly_averages[month] # 18 rows with four averages each in the current data
        headers = monthly_headers[month] # this month's four parameter names
        data = monthly_values[month]

        #convert data to np.array type
        #because that is the format for the gmm clustering function
        data = np.array(data)
        dims = data.shape[1]


        datapt_groups, means, weights, covariances, details = gaussian_clusters(data, distros, dims, err_tolerance)


        if not datapt_groups:
            return # stop if clustering reported an error

        try:
            write_output(month, regions, averages, data, headers, datapt_groups, means, details, distros) #still needs to be fixed
        except OSError:
            print(f"Error: Cannot save {month}'s results. Check whether the output folder is writable or a file is open.")
            return

        file = OUTPUT_FOLDER / month / "clusters.png"
        try:
            draw_clusters(regions, averages, headers, datapt_groups, file, month)
        except OSError:
            print(f"Error: Could not update {month}'s cluster plot.")
            return
            

    return

if __name__ == "__main__":
    main()
