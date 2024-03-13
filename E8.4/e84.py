def memorize_multi_class(data, labels, num_classes):
    """
    Calculate the upper-bound memory-equivalent capacity (mec) for multi-class classification.

    Args:
        data (list): List of d-dimensional vectors (n x d).
        labels (list): List of labels (0 to num_classes-1) with length n.
        num_classes (int): Number of classes (k).

    Returns:
        float: The calculated mec.
    """
    thresholds = 0
    table = []
    sortedtable = []
    class_id = 0

    # Create a table with (Σx[i]*d[i], label[i]) for each data point
    for i in range(len(data)):
        table.append((sum(data[i]), labels[i]))

    # Sort the table based on the first column (Σx[i]*d[i])
    sortedtable = sorted(table, key=lambda x: x[0])

    # Determine the thresholds and update class_id
    for i in range(len(sortedtable)):
        if sortedtable[i][1] != class_id:
            class_id = sortedtable[i][1]
            thresholds += 1

    # Calculate the minimum number of thresholds
    minthreshs = log2(thresholds + 1)

    # Calculate mec
    mec = (minthreshs * (len(data[0]) + num_classes)) + (minthreshs + num_classes)

    return mec


# Example usage
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example data (n x d)
labels = [0, 1, 2]  # Example labels (0 to num_classes-1)
num_classes = 3

mec_result = memorize_multi_class(data, labels, num_classes)
print(f"Multi-class mec: {mec_result}")

import numpy as np
from math import log2


def memorize_regression(data, labels):
    """
    Calculate the upper-bound memory-equivalent capacity (mec) for regression.

    Args:
        data (list): List of d-dimensional vectors (n x d).
        labels (list): List of regression labels with length n.

    Returns:
        float: The calculated mec.
    """
    thresholds = 0
    table = []
    sortedtable = []
    class_id = None

    # Create a table with (Σx[i]*d[i], label[i]) for each data point
    for i in range(len(data)):
        table.append((sum(data[i]), labels[i]))

    # Sort the table based on the first column (Σx[i]*d[i])
    sortedtable = sorted(table, key=lambda x: x[0])

    # Determine the thresholds and update class_id
    for i in range(len(sortedtable)):
        if sortedtable[i][1] != class_id:
            class_id = sortedtable[i][1]
            thresholds += 1

    # Calculate the minimum number of thresholds
    minthreshs = log2(thresholds + 1)

    # Calculate mec
    mec = (minthreshs * len(data[0])) + minthreshs

    return mec


# Example usage
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
labels = [10, 20, 30]
mec = memorize_regression(data, labels)
print(f"The memory-equivalent capacity for regression is {mec:.2f}")
