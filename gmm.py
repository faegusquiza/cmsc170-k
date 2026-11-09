import csv
import numpy as np
import math
import random
import csv
from read_data import MONTHS, read_monthly_data, standardize_data
from pathlib import Path

START_YEAR = 2001
END_YEAR = 2025
OUTPUT_FOLDER = Path("results/gaussian/monthly")

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


    A = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
    
    B = np.array([[20,67,69],
                 [420,11,4],
                 [16,23,13]])


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

def gaussian_clusters(data, distros, dims, err_tolerance):

    #initialize the means such that it takes k random samples from our data
    means = []
    while len(means) < k:
        random_index = random.choice(data)
        if random_index not in means: # skip a region that is already a centroid
            means.append(random_index)
    means = np.array(means)
    
    # Probability of all detaset towards the k clusters
    # This will be a list, where the elements is a list with k elements
    # THIS IS A NUMPY ARRAY
    probabilities = np.zeros((len(data), distros))

    #initialize the weights by assuming all clusters are equally likely using uniform distribution
    weights = [1/distros]*distros
    weights = np.array(weights)

    #Initialize covariance matrices
    cov_mat = [gen_global_cov(data=data, dimensions=dims) for _ in range(distros) ] 
    cov_mat = np.array(cov_mat)

    #initiate log_likelihood to infinity
    log_like_old = math.inf

    datapt_group = [[] for _ in range(distros)]
    
    # We now have our needed data...
    # we do the looping!
    while(1):
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
        datapt_group = [[] for _ in range(distros)]
        for j in range(len(data)):
            max = -math.inf
            max_idx = math.inf
            for l in range(k):
                if probabilities[j][l] > max:
                    max = probabilities[j][l]
                    max_idx = l
            # print(probabilities[j])
            # print("chosen index:", max_idx, "which has max:", max)
            datapt_group[max_idx].append(data[j])



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

            cov_mat[k] = cov_numerator / mean_and_cov_denom_and_weigth_sum

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
            print("convergence detected")
            break
        log_like_old = log_like_new 

    return datapt_group

# def write_output(month, regions, averages, climate_values, headers, labels, centroids, details, k):
#     folder = OUTPUT_FOLDER / month # e.g. results/gaussian/monthly/{month}
#     folder.mkdir(parents=True, exist_ok=True) # create this month's folder if needed

#     # clusters
#     f = open(folder / "clusters.csv", "w", newline="", encoding="utf-8")
#     writer = csv.writer(f)
#     writer.writerow(["REGION", "CLUSTER"])
#     for i in range(len(regions)):
#         writer.writerow([regions[i], labels[i]])
#     f.close()

#     # climate averages
#     f = open(folder / "climate_averages.csv", "w", newline="", encoding="utf-8")
#     writer = csv.writer(f)
#     writer.writerow(["REGION"] + headers)
#     for i in range(len(regions)):
#         writer.writerow([regions[i]] + averages[i])
#     f.close()

#     # standardized climate values
#     f = open(folder / "standardized_values.csv", "w", newline="", encoding="utf-8")
#     writer = csv.writer(f)
#     writer.writerow(["REGION"] + headers)
#     for i in range(len(regions)):
#         writer.writerow([regions[i]] + climate_values[i])
#     f.close()

#     # final centroids in standardized values (what K-means used)
#     f = open(folder / "centroids.csv", "w", newline="", encoding="utf-8")
#     writer = csv.writer(f)
#     writer.writerow(["CLUSTER"] + headers)
#     for i in range(len(centroids)):
#         writer.writerow([i + 1] + centroids[i])
#     f.close()

#     # each cluster's average in the original units (for interpreting the climate groups)
#     f = open(folder / "cluster_averages.csv", "w", newline="", encoding="utf-8")
#     writer = csv.writer(f)
#     writer.writerow(["CLUSTER"] + headers)
#     profiles = cluster_averages(averages, labels)
#     for i in range(len(profiles)):
#         writer.writerow([i + 1] + profiles[i])
#     f.close()

#     # summary of clusters and their members
#     starting_regions = [regions[index] for index in details["initial_indices"]]
#     lines = [f"{month} ({START_YEAR}-{END_YEAR}): Gaussian Mixture Model Clustering"]
#     lines.append(f"k = {k}")
#     lines.append(f"Initial centroids: {', '.join(starting_regions)}")
#     lines.append(f"Iterations: {details['iterations']}")
#     for label in range(1, max(labels) + 1):
#         members = []
#         for i in range(len(regions)):
#             if labels[i] == label:
#                 members.append(regions[i])
#         lines.append(f"Cluster {label}: {', '.join(members)}")

#     summary = "\n".join(lines)
#     print("\n" + summary)

#     f = open(folder / "summary.txt", "w", encoding="utf-8")
#     f.write(summary + "\n")
#     f.close()

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


def main():

    regions, monthly_values, monthly_headers, monthly_averages = read_monthly_data()

    print("Gaussion Mixture Model Clustring for the climate of the regions in the Philippines.")
    print(f"Regions read: {len(regions)}")
    print(f"Monthly datasets: {len(MONTHS)}\nParameters per region per month: {len(monthly_headers['JAN'])}\nYears averaged: {START_YEAR}-{END_YEAR}\n")

    distros = get_k(len(regions))
    print("Enter the convergence error tolerance (e.g. 0.00001):")
    err_tolerance = float(input())

    for month in MONTHS:

        print(monthly_averages)



        # dims = 2
        # err_tolerance = 0.0001

        #INITIALIZATIONS
        return



            

    return

if __name__ == "__main__":
    main()
