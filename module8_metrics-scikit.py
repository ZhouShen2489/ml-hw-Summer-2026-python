import numpy as np
from sklearn.metrics import precision_score, recall_score


class ClassificationMetrics:

    # Step 0: Initialize the class with N
    def __init__(self, N):
        self.N = N
        self.points = np.zeros((N, 2), dtype=int)

    # Step 1: Insert the points
    def input_points(self):
        for i in range(self.N):
            print("Point:", i + 1)

            x = int(input("Enter the x value (ground truth class, 0 or 1): "))
            y = int(input("Enter the y value (predicted class, 0 or 1): "))

            self.points[i] = [x, y]

    # Step 2: Calculate precision and recall
    def calculate_metrics(self):
        # Separate the ground truth labels and predicted labels
        X_values = self.points[:, 0]
        y_values = self.points[:, 1]

        # Use Scikit-learn to calculate the metrics
        precision = precision_score(X_values, y_values)
        recall = recall_score(X_values, y_values)

        return precision, recall


# Number of points
N = int(input("Enter the number of points N: "))

if N <= 0:
    print("Error: N must be a positive integer.")

else:
    metrics = ClassificationMetrics(N)

    # Input ground truth and predicted classes
    metrics.input_points()

    # Calculate precision and recall
    precision, recall = metrics.calculate_metrics()

    print("Precision of these points:", precision)
    print("Recall of these points:", recall)
