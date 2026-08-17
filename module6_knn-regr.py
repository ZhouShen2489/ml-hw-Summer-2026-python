import numpy as np


class KNNRegression:

    def __init__(self, N, k):
        self.N = N
        self.k = k
        self.points = np.zeros((N, 2))

    # Step 1: Insert the points
    def input_points(self):
        for i in range(self.N):
            print("Point:", i + 1)

            x = float(input("Enter the x-coordinate: "))
            y = float(input("Enter the y-coordinate: "))

            self.points[i] = [x, y]

    # Step 2: Predict y value for X
    def predict(self, X):

        # Initialize distance and y-coordinate array
        neighbors_distances = np.zeros((self.N, 2))

        for i in range(self.N):

            # Calculate Manhattan distance 
            distance = np.sqrt((self.points[i][0] - X) ** 2)

            neighbors_distances[i, 0] = distance
            neighbors_distances[i, 1] = self.points[i][1]

        # Sort the neighbors by distance
        neighbors_distances = neighbors_distances[
            neighbors_distances[:, 0].argsort()
        ]

        # Take k nearest neighbors' y-coordinates
        k_nearest_neighbors_y = neighbors_distances[:self.k, 1]

        # Average their y values
        X_y_predict = k_nearest_neighbors_y.mean()

        return X_y_predict


# Number of points
N = int(input("Enter the number of points N: "))

# Number of nearest neighbors
k = int(input("Enter the number of nearest neighbors k: "))


if k > N or k <= 0:
    print("Error: k is not a valid number of nearest neighbors.")

else:
    # Create the k-NN Regression object
    knn = KNNRegression(N, k)

    # Input training points
    knn.input_points()

    # Input X to predict
    X = float(
        input("Enter the x-coordinate of the point you want to predict: ")
    )

    # Predict Y
    Y = knn.predict(X)

    print("The predicted y-coordinate of the point with x: ", X,"is: ",Y)