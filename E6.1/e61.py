import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    print('GNN = (#correct predictions) / (#training points)')
    print('Multi-Class Classification (5 classes), repeat 30 times:')
    for _ in range(30):
        # Generate synthetic data (you can replace this with your own dataset)
        X, y = make_classification(
            n_samples=10000, n_features=50, n_redundant=0, n_classes=3, n_informative=49
        )

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.69)

        # Initialize KNN classifier with k=3 (you can choose a different k)
        knn_model = KNeighborsClassifier(n_neighbors=1000)

        # Fit the model to the training data
        knn_model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = knn_model.predict(X_test)

        # Calculate accuracy
        acc = accuracy_score(y_test, y_pred)
        # print(f"Accuracy: {acc:.2f}")

        gnn = acc * y_test.shape[0] / X_train.shape[0]
        print(f'GNN: {gnn}')
