import warnings
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    train_test_split,
    TimeSeriesSplit,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Logistic Regression as an example model
model = LogisticRegression()

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_accuracies = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    kf_accuracies.append(accuracy)

print(f"K-Fold Cross-Validation Accuracies: {kf_accuracies}")