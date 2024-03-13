from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# iris = load_breast_cancer()
# X = iris.data[iris.target < 2]
# y = iris.target[iris.target < 2]

# Load your binary dataset and preprocess it (e.g., handle missing values, encoding)
X, y = make_classification(
    n_samples=1000, n_features=50, n_redundant=0, n_classes=2, n_informative=49
)

clf = DecisionTreeClassifier()

print("Train on the Artificial Dataset")

# Strategy 1: Generate one clause per row
clauses = []
for i in range(len(X)):
    clause = f"if x1={X[i,0]} and x2={X[i,1]}: return {y[i]}"
    clauses.append(clause)
print("Number of clauses for strategy 1 (Generate one clause per row):", len(clauses))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Strategy 2: Consolidate on single feature
clauses = []
for c in [0, 1]:
    clause = f"if x1>=mean(x1): return {c}"
    clauses.append(clause)
print("Number of clauses for strategy 2 (Consolidate on single feature):", len(clauses))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Strategy 3: Decision tree pruning
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier().fit(X, y)
print(
    "Number of clauses for strategy 3 (Decision tree pruning):", tree.tree_.node_count
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
