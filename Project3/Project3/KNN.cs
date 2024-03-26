using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project3
{
    internal class KNN
    {

        #region 1
        /*
         K-Nearest Neighbors (KNN) is a simple and versatile machine learning algorithm used for both classification and regression tasks. It's a non-parametric and instance-based learning algorithm, meaning it doesn't make assumptions about the underlying data distribution and learns directly from the training instances. In KNN, predictions are made based on the majority class or the average value of the K nearest neighbors in the feature space.

Here's an explanation of how KNN works:

**1. Training Phase:**

- During the training phase, KNN stores all available data points and their corresponding class labels or response values in memory. This step involves only storing the data and does not involve any computation.

**2. Prediction Phase:**

- To make predictions for a new data point, KNN calculates the distance between the new data point and every point in the training dataset. The distance metric used (e.g., Euclidean distance, Manhattan distance, etc.) depends on the problem and data characteristics.

- KNN then selects the K nearest neighbors (data points) to the new data point based on the calculated distances. K is a hyperparameter that is typically chosen based on cross-validation or domain knowledge.

- For classification tasks, KNN assigns the class label that is most frequent among the K nearest neighbors to the new data point. This is often determined by a majority voting mechanism.

- For regression tasks, KNN predicts the response value for the new data point by averaging the response values of its K nearest neighbors.

**3. Hyperparameter Tuning:**

- The choice of K is a critical hyperparameter in KNN. A small value of K can lead to high variance and overfitting, while a large value of K can lead to high bias and underfitting. Therefore, the value of K needs to be carefully selected based on the problem and data characteristics.

- Additionally, the choice of distance metric and any other hyperparameters (e.g., weights assigned to neighbors) can significantly impact the performance of the KNN algorithm.

**4. Scalability and Efficiency:**

- One of the main drawbacks of KNN is its computational inefficiency, especially with large datasets, as it requires computing distances to every point in the dataset during prediction.

- Approximate nearest neighbor search techniques (e.g., KD-trees, ball trees) can be used to speed up the prediction process, but they come with their own trade-offs and limitations.

**5. Applications:**

- KNN is commonly used in classification tasks such as image recognition, document categorization, and recommendation systems.
  
- It's also used in regression tasks such as predicting house prices based on similar properties, forecasting stock prices, and estimating missing values in datasets.

In summary, KNN is a simple yet powerful algorithm that relies on the principle of similarity to make predictions. It's easy to understand, implement, and interpret, making it a popular choice for various machine learning tasks, especially in scenarios where the decision boundary is complex or not easily characterized by parametric models. However, its scalability and computational efficiency can be limitations, particularly with large datasets.
         */
        #endregion 1

        #region 2
        /*
         The mathematics behind the K-Nearest Neighbors (KNN) algorithm mainly involves calculating distances between data points in the feature space to determine the nearest neighbors. Let's discuss the key mathematical concepts involved:

**1. Distance Metrics:**

KNN typically uses distance metrics to measure the similarity or dissimilarity between data points. The most commonly used distance metrics are:

- **Euclidean Distance**:
  \[ d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2} \]

- **Manhattan Distance (City Block Distance)**:
  \[ d(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^{n} |p_i - q_i| \]

- **Minkowski Distance**:
  \[ d(\mathbf{p}, \mathbf{q}) = \left( \sum_{i=1}^{n} |p_i - q_i|^p \right)^{\frac{1}{p}} \]

Where \( \mathbf{p} \) and \( \mathbf{q} \) are two data points, and \( p_i \) and \( q_i \) are their respective coordinates along each feature dimension.

**2. Finding Nearest Neighbors:**

Given a new data point \( \mathbf{x} \), KNN identifies the \( K \) nearest neighbors to \( \mathbf{x} \) from the training dataset based on the chosen distance metric. It computes the distance between \( \mathbf{x} \) and each data point in the training dataset, then selects the \( K \) data points with the smallest distances.

**3. Majority Voting (Classification) or Averaging (Regression):**

For classification tasks, once the \( K \) nearest neighbors are identified, KNN predicts the class label of the new data point based on the majority class among its neighbors. For regression tasks, it predicts the response value by averaging the response values of its \( K \) nearest neighbors.

**4. Weighted KNN:**

In some cases, it may be beneficial to assign different weights to the nearest neighbors based on their distances. Weighted KNN assigns higher weights to closer neighbors and lower weights to farther neighbors. This is typically done by inversely proportional weighting, where the weight of a neighbor is inversely proportional to its distance from the query point.

**5. Choosing the Value of \( K \):**

The choice of \( K \) is crucial in KNN. A smaller value of \( K \) results in more complex decision boundaries and can lead to overfitting, while a larger value of \( K \) results in smoother decision boundaries and can lead to underfitting. The value of \( K \) is typically chosen using techniques such as cross-validation.

**6. Scalability and Computational Complexity:**

KNN's computational complexity grows linearly with the size of the training dataset since it requires computing distances to every point during prediction. This can be computationally expensive for large datasets, especially in high-dimensional spaces.

In summary, the mathematics of KNN involves calculating distances between data points, identifying the nearest neighbors, and making predictions based on their class labels or response values. The choice of distance metric, value of \( K \), and any additional considerations such as weighted KNN are essential aspects of implementing and understanding the algorithm.
         */
        #endregion 2

        #region 3
        /*
         Let's demonstrate how the K-Nearest Neighbors (KNN) algorithm works with a simple example using Python. In this example, we'll use the famous Iris dataset, which contains measurements of various iris flowers, and we'll classify the species of iris based on their sepal length and width.

```python
# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length and width)
y = iris.target  # Target (species)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for distance-based algorithms like KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the KNN classifier
k = 5  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

Explanation of the steps:

1. We import necessary libraries including `load_iris` to load the Iris dataset, `train_test_split` to split the data into training and test sets, `StandardScaler` to standardize the features, `KNeighborsClassifier` to create the KNN classifier, and `accuracy_score` to calculate accuracy.

2. We load the Iris dataset and separate the features (sepal length and width) and the target (species).

3. We split the dataset into training and test sets, with 70% of the data used for training and 30% for testing.

4. We standardize the features using `StandardScaler` to ensure that all features have the same scale, which is important for distance-based algorithms like KNN.

5. We create the KNN classifier with 5 neighbors (`k=5`) and train it using the training data.

6. We make predictions on the test set using the trained classifier.

7. Finally, we calculate the accuracy of the model by comparing the predicted labels (`y_pred`) with the true labels (`y_test`).

The output will display the accuracy of the KNN classifier on the test set, indicating how well the model performs in classifying iris species based on their sepal measurements.
         */
        #endregion 3
    }
}
