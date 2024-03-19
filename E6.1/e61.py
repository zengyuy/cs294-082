import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    dim = [2, 4, 8, 16]
    # Initialize KNN classifier
    knn_model = KNeighborsClassifier(n_neighbors=1)
    print("Binary Classification:")
    res = []
    for d in dim:
        # Generate synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=d,
            n_informative=d,
            n_redundant=0,
            n_classes=2,
            n_clusters_per_class=1,
        )

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.996)

        # Fit the model to the training data
        knn_model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = knn_model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        # print(f"Accuracy: {acc:.2f}")

        avg_mem_size = acc * 2**d
        print(
            f"d={d}: n_full={2**d}, Avg. req. points for memorization n_avg={avg_mem_size:.2f}, n_full/n_avg={(2**d)/avg_mem_size}"
        )

        res.append((2**d) / avg_mem_size)
