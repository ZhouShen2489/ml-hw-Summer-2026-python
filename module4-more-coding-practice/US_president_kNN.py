import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)


# --------------------------------------------------
# Step 1: Read the dataset
# --------------------------------------------------

# Option A: Read directly from the original online dataset
url = (
    "https://raw.githubusercontent.com/deepanshu88/Datasets/"
    "master/UploadedFiles2/US%20Presidential%20Data.csv"
)

data1 = pd.read_csv(url)

# Option B: If you downloaded the CSV into the same folder,
# comment out Option A and use:
# data1 = pd.read_csv("US Presidential Data.csv")


# Display the data
print(data1.head())
print("\nDataset shape:", data1.shape)
print("\nColumn names:")
print(data1.columns.tolist())


# --------------------------------------------------
# Step 2: Separate predictors and target
# --------------------------------------------------

# The actual CSV column is named "Win/Loss", not "Win.Loss"
X = data1.drop(columns=["Win/Loss"])
y = data1["Win/Loss"]


# --------------------------------------------------
# Step 3: Split into training and validation sets
# --------------------------------------------------

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=101,
    stratify=y,
)

print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_validation.shape)


# --------------------------------------------------
# Step 4: Build a pipeline
# --------------------------------------------------
# StandardScaler is important because KNN uses distances.
# It prevents variables with large numeric ranges from dominating.

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ]
)


# --------------------------------------------------
# Step 5: Use repeated 10-fold cross-validation
# to find the best k
# --------------------------------------------------

cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=3,
    random_state=1234,
)

k_values = list(range(5, 25, 2))

parameter_grid = {
    "knn__n_neighbors": k_values
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)

grid_search.fit(X_train, y_train)


# --------------------------------------------------
# Step 6: Show the best k
# --------------------------------------------------

print("\nBest K:", grid_search.best_params_["knn__n_neighbors"])
print("Best cross-validation ROC AUC:", grid_search.best_score_)


# Show results for every tested k
results = pd.DataFrame(grid_search.cv_results_)

summary = results[
    [
        "param_knn__n_neighbors",
        "mean_test_score",
        "std_test_score",
    ]
].copy()

summary.columns = [
    "K",
    "Mean CV ROC AUC",
    "Standard Deviation",
]

print("\nCross-validation results:")
print(summary)


# --------------------------------------------------
# Step 7: Evaluate on validation data
# --------------------------------------------------

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_validation)
probabilities = best_model.predict_proba(X_validation)[:, 1]

accuracy = accuracy_score(y_validation, predictions)
auc = roc_auc_score(y_validation, probabilities)

print("\nValidation accuracy:", accuracy)
print("Validation ROC AUC:", auc)

print("\nConfusion matrix:")
print(confusion_matrix(y_validation, predictions))

print("\nClassification report:")
print(classification_report(y_validation, predictions))


# --------------------------------------------------
# Step 8: Plot ROC curve
# --------------------------------------------------

RocCurveDisplay.from_predictions(
    y_validation,
    probabilities,
)

plt.title("KNN ROC Curve — US Presidential Data")
plt.show()


# --------------------------------------------------
# Step 9: Plot K versus cross-validation ROC AUC
# --------------------------------------------------

plt.figure()

plt.plot(
    summary["K"].astype(int),
    summary["Mean CV ROC AUC"],
    marker="o",
)

plt.xlabel("Number of neighbors, K")
plt.ylabel("Mean cross-validation ROC AUC")
plt.title("Selecting the Best K")
plt.xticks(summary["K"].astype(int))
plt.show()