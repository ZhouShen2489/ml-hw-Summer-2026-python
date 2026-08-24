import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class KNNRegression:

    # Step 0: Initialize the class with N and k
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
        # Separate the x value and y value
        X_values = self.points[:, 0].reshape(-1, 1)
        y_values = self.points[:, 1]

        # Use Scikit-learn 
        model = KNeighborsRegressor(n_neighbors=self.k)
        model.fit(X_values, y_values)

        # Predict the y value for the given X
        predicted_y = model.predict(np.array([[X]]))

        return predicted_y[0]

    # Calculate the variance of the labels
    def get_variance(self):
        return np.var(self.points[:, 1])


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

    # Show the variance of labels
    print("Variance of labels in the training data:", knn.get_variance())

    # Input X to predict
    X = float(
        input("Enter the x-coordinate of the point you want to predict: ")
    )

    # Predict Y
    Y = knn.predict(X)

    print("The predicted y-coordinate of the point with x: ", X, "is: ", Y)
