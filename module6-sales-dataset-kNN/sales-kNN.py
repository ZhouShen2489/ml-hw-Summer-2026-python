"""KNN regression example based on the Analytics Vidhya tutorial."""

from math import sqrt
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = Path(__file__).resolve().parent
TARGET = "Item_Outlet_Sales"


def clean_features(frame, item_weight_mean, outlet_size_mode):
    frame = frame.copy()
    frame["Item_Weight"] = frame["Item_Weight"].fillna(item_weight_mean)
    frame["Outlet_Size"] = frame["Outlet_Size"].fillna(outlet_size_mode)
    return frame.drop(["Item_Identifier", "Outlet_Identifier"], axis=1)


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(train_df.head(5))

    item_weight_mean = train_df["Item_Weight"].mean()
    outlet_size_mode = train_df["Outlet_Size"].mode()[0]
    train_features = clean_features(train_df.drop(TARGET, axis=1), item_weight_mean, outlet_size_mode)
    test_features = clean_features(test_df, item_weight_mean, outlet_size_mode)

    train_features = pd.get_dummies(train_features)
    test_features = pd.get_dummies(test_features)
    test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

    x_train, x_valid, y_train, y_valid = train_test_split(
        train_features, train_df[TARGET], test_size=0.3, random_state=42
    )
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)

    rmse_values = []
    for k in range(1, 21):
        model = neighbors.KNeighborsRegressor(n_neighbors=k)
        model.fit(x_train, y_train)
        prediction = model.predict(x_valid)
        error = sqrt(mean_squared_error(y_valid, prediction))
        rmse_values.append(error)
        print("RMSE value for k =", k, "is:", error)

    best_k = rmse_values.index(min(rmse_values)) + 1
    print("Best k:", best_k)
    print("Best RMSE:", min(rmse_values))

    # Use cross-validation to select k, following the tutorial's GridSearchCV step.
    parameter_grid = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9]}
    grid_search = GridSearchCV(
        neighbors.KNeighborsRegressor(), parameter_grid, cv=5, scoring="neg_root_mean_squared_error"
    )
    grid_search.fit(x_train, y_train)
    grid_k = grid_search.best_params_["n_neighbors"]
    print("Grid search best k:", grid_k)
    print("Grid search best RMSE:", -grid_search.best_score_)

    # Plot and save the elbow curve for viewing in VS Code.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 21), rmse_values, marker="o")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("RMSE")
    plt.title("KNN Regression: RMSE by k")
    plt.grid(True)
    plt.tight_layout()
    plot_path = DATA_DIR / "knn-rmse-plot.png"
    plt.savefig(plot_path)
    plt.show()
    print("Saved RMSE plot to", plot_path)

    # Fit the selected model on all labeled training data and predict test.csv.
    final_scaler = MinMaxScaler(feature_range=(0, 1))
    all_train = final_scaler.fit_transform(train_features)
    all_test = final_scaler.transform(test_features)
    final_model = neighbors.KNeighborsRegressor(n_neighbors=grid_k)
    final_model.fit(all_train, train_df[TARGET])
    test_prediction = final_model.predict(all_test)

    submission = test_df[["Item_Identifier", "Outlet_Identifier"]].copy()
    submission[TARGET] = test_prediction
    output_path = DATA_DIR / "submit_file.csv"
    submission.to_csv(output_path, index=False)
    print("Saved test predictions to", output_path)


if __name__ == "__main__":
    main()
