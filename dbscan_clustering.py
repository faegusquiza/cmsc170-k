import math
from pathlib import Path
import matplotlib.pyplot as plt

from read_data import MONTHS, read_monthly_data, standardize_data

OUTPUT_FOLDER = Path("results/dbscan")

# DBSCAN Settings
MINPTS = 5

def main():
    regions, monthly_values, monthly_headers, monthly_averages = read_monthly_data()

    epsilon = calculate_epsilon(monthly_averages["JAN"])

def calculate_epsilon(data_points):
    # Calculate for the K Distance
    k_distance = get_k_distance(data_points)

    # graph_k_distance(k_distance)

    # Find a function that best fits the matrix
    x = range(len(k_distance))
    degree, coefficients = find_best_fit_regression(x, k_distance, max_degree=6)
    y = solve(x, coefficients)

    # Graph regression function
    # graph_regression(f"{degree}-degree Regression", x, y, k_distance)
    # graph_function(f"{degree}-degree Regression", x, y)

    # Find the index of the knee using kneedle algorithm
    knee = kneedle(x, y)

    epsilon = k_distance[knee]

    return k_distance

def kneedle(x: list[float], y: list[float]):
    if len(x) != len(y):
        raise ArithmeticError("x and y must have the same length")

    # Normalization
    x_normalized = normalize(x)
    y_normalized = normalize(y)

    # Difference
    y_difference = [x_normalized[i] - y_normalized[i] for i in range(len(x))]

    # Local Maxima
    y_local_maxima = [y_difference[i] for i in range(1, len(y_difference)-1) if y_difference[i-1] < y_difference[i] and y_difference[i+1] < y_difference[i]]

    return y_difference.index(y_local_maxima[0])

def normalize(l: list[float]):
    l_max = max(l)
    l_min = min(l)
    denominator = l_max - l_min

    l_normalized = []

    for value in l:
        l_normalized.append((value-l_min)/denominator)

    return l_normalized


def get_k_distance(data_points: list[float]):
    distance_matrix = get_distance_matrix(data_points)

    distance_matrix.sort()

    return distance_matrix

def get_curvatures(x: list[float], coefficients: list[float]):
    first_derivative = get_differential_coefficients(coefficients)
    second_derivative = get_differential_coefficients(first_derivative)

    k = []

    first_derivative_x = solve(x, first_derivative)
    second_derivative_x = solve(x, second_derivative)

    for i in range(len(x)):
        curvature = second_derivative_x[i]/((1 + first_derivative_x[i] ** 2) ** (3/2))

        k.append(curvature)

    return k

def get_differential_coefficients(coefficients):
    differential_coefficients = coefficients[:]

    for degree, coefficient in enumerate(coefficients):
        differential_coefficients[degree] *= max(0, degree-1)

    differential_coefficients += [0]

    return differential_coefficients[1:]

    
def find_best_fit_regression(x: list[float], y: list[float], max_degree = 10):
    root_mean_squared_errors = []

    for degree in range(max_degree):
        best_fit_coefficients = polynomial_regression(x, y, degree)

        predicted_y = solve(x, best_fit_coefficients)

        root_mean_squared_errors.append(calculate_root_mean_squared_error(predicted_y, y))

    best_fit_degree = root_mean_squared_errors.index(min(root_mean_squared_errors))
    best_fit_coefficients = polynomial_regression(x, y, best_fit_degree)
    
    return best_fit_degree, best_fit_coefficients 

def calculate_root_mean_squared_error(predicted_y, actual_y):
    total_error = 0
    data_points_count = len(actual_y)

    for i in range(data_points_count):
        total_error += (actual_y[i] - predicted_y[i]) ** 2

    return (total_error/data_points_count) ** 0.5
    

def generate_features(x, degree):
    """
    Generate polynomial features for the given x values.

    Returns a matrix containing:
    [1, x, x^2, ..., x^degree]
    """
    features = []

    for value in x:
        row = []

        for power in range(degree + 1):
            row.append(value ** power)

        features.append(row)

    return features

def transpose(matrix):
    return [
        [matrix[row][col] for row in range(len(matrix))]
        for col in range(len(matrix[0]))
    ]

def matrix_multiply(A, B):
    result = []

    for i in range(len(A)):
        row = []

        for j in range(len(B[0])):
            total = 0

            for k in range(len(B)):
                total += A[i][k] * B[k][j]

            row.append(total)

        result.append(row)

    return result

def gaussian_elimination(A, B):
    n = len(A)

    # Create augmented matrix
    matrix = [
        A[i][:] + B[i][:]
        for i in range(n)
    ]

    # Forward elimination
    for i in range(n):

        # Partial pivoting
        max_row = i

        for j in range(i + 1, n):
            if abs(matrix[j][i]) > abs(matrix[max_row][i]):
                max_row = j

        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]

        pivot = matrix[i][i]

        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned.")

        for j in range(i + 1, n):
            factor = matrix[j][i] / pivot

            for k in range(i, n + 1):
                matrix[j][k] -= factor * matrix[i][k]

    # Back substitution
    coefficients = [0] * n

    for i in range(n - 1, -1, -1):
        total = matrix[i][n]

        for j in range(i + 1, n):
            total -= matrix[i][j] * coefficients[j]

        coefficients[i] = total / matrix[i][i]

    return coefficients

def solve(x: list[float], coefficients: list[float]):
    predictions = []

    for value in x:
        result = 0

        for power, coefficient in enumerate(coefficients):
            result += coefficient * (value ** power)

        predictions.append(result)

    return predictions

def polynomial_regression(x, y, degree):
    X = generate_features(x, degree)

    XT = transpose(X)

    XT_X = matrix_multiply(XT, X)

    # Convert y into a column matrix
    Y = [[value] for value in y]

    XT_Y = matrix_multiply(XT, Y)

    coefficients = gaussian_elimination(XT_X, XT_Y)

    return coefficients

def graph_function(title, x, y):
    plt.plot(x, y)

    plt.title(title)

    plt.show()

def graph_regression(title, x, y, data_points):
    plt.plot(x, y)
    plt.scatter(x, data_points)

    plt.title(title)

    plt.show()

def graph_k_distance(k_distance):
    plt.plot(range(len(k_distance)), k_distance, linestyle='-')

    plt.title("K-distance")
    plt.xlabel("No. of Paired Points")
    plt.ylabel("Distance")

    plt.show()

def get_distance_matrix(data_points):
    distance_matrix = []

    # Get the Euclidian distance for each data points
    for data_point_a in data_points:
        for data_point_b in data_points:
            # Skip similar data points
            if (data_point_a == data_point_b):
                continue

            distance = get_euclidian_distance(data_point_a, data_point_b)
            distance_matrix.append(distance)

    return distance_matrix

def get_euclidian_distance(a, b):
    if (len(a) != len(b)):
        raise ArithmeticError(f"{a} and {b} must have the same length")
    
    distance = 0

    for i in range(len(a)):
        distance += (a[i] - b[i]) ** 2

    distance **= 0.5

    return distance

main()
