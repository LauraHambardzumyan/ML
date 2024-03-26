using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project3
{
    internal class SVM
    {
        #region
        /*
         Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks. It's particularly well-suited for classification of complex datasets where the decision boundary is non-linear or not easily separable. SVM aims to find the optimal hyperplane that best separates the classes in the feature space while maximizing the margin between the classes. Let's delve into the explanation of how SVM works:

**1. Linear SVM for Binary Classification:**

- In a binary classification problem, SVM aims to find the hyperplane that separates the two classes with the maximum margin. This hyperplane is defined by the equation:
   \[ \mathbf{w} \cdot \mathbf{x} + b = 0 \]
   where \( \mathbf{w} \) is the normal vector to the hyperplane, \( \mathbf{x} \) is the input feature vector, and \( b \) is the bias term.

- The distance between the hyperplane and the closest data points from each class is known as the margin. SVM maximizes this margin, ensuring robustness and generalizability of the model.

- The data points closest to the hyperplane, which determine the margin, are called support vectors. These are the critical points for defining the decision boundary.

**2. Non-Linear SVM using Kernels:**

- In cases where the data is not linearly separable, SVM can still be applied by using kernel functions. Kernel functions transform the input features into a higher-dimensional space where the data may become linearly separable.

- Popular kernel functions include:
  - **Linear Kernel**: \( K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j \)
  - **Polynomial Kernel**: \( K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i \cdot \mathbf{x}_j + r)^d \)
  - **Radial Basis Function (RBF) Kernel**: \( K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma ||\mathbf{x}_i - \mathbf{x}_j||^2) \)

- These kernel functions map the input features into a higher-dimensional space where the classes might become linearly separable. The decision boundary is then constructed in this higher-dimensional space.

**3. Soft Margin SVM:**

- In real-world scenarios, datasets may contain noise or overlapping classes, making them not entirely separable. Soft margin SVM allows for some misclassifications by introducing a penalty term for misclassified data points.

- The objective is to find the hyperplane that separates the classes with the maximum margin while minimizing the classification error. The trade-off between maximizing the margin and minimizing the error is controlled by a regularization parameter \( C \).

**4. Multi-Class Classification:**

- SVM inherently supports binary classification. For multi-class classification problems, SVM can be extended using strategies like:
  - One-vs-One (OvO): Build multiple binary classifiers for each pair of classes and combine their predictions.
  - One-vs-All (OvA): Build one binary classifier for each class, treating it as the positive class and the rest as the negative class.

**5. Applications:**

- SVM is widely used in various domains such as text classification, image recognition, bioinformatics, and finance for tasks like sentiment analysis, object detection, gene expression classification, and credit scoring.

In summary, SVM is a versatile and effective algorithm for classification tasks, capable of handling complex datasets and achieving high accuracy by finding optimal decision boundaries. Its ability to handle non-linear data through kernel tricks makes it one of the most popular and widely used machine learning algorithms.
         */
        #endregion

        #region
        /*
         Support Vector Machine (SVM) is a powerful supervised learning algorithm used for classification and regression tasks. SVM aims to find the optimal hyperplane that separates different classes in the feature space while maximizing the margin between the classes. In this explanation, I'll focus on the mathematics behind SVM for binary classification.

**1. Decision Boundary:**

In SVM, given a set of labeled training data, each belonging to one of two classes (positive and negative), the algorithm finds the hyperplane that best separates the classes. The hyperplane is represented by the equation:

\[ \mathbf{w}^T \mathbf{x} + b = 0 \]

where:
- \( \mathbf{w} \) is the weight vector perpendicular to the hyperplane,
- \( \mathbf{x} \) is the feature vector,
- \( b \) is the bias term.

The decision boundary is the hyperplane that separates the classes, with points on one side belonging to one class and points on the other side belonging to the other class.

**2. Margin:**

The margin is the distance between the decision boundary and the nearest data point from either class. SVM aims to maximize this margin, as it helps in improving the generalization of the model.

Mathematically, the margin can be calculated as the distance between the decision boundary and a support vector (a data point closest to the decision boundary), which can be expressed as:

\[ \text{margin} = \frac{1}{\|\mathbf{w}\|} \]

**3. Optimal Hyperplane:**

In SVM, the optimal hyperplane is the one that maximizes the margin. This optimization problem can be formulated as a constrained optimization problem:

\[ \text{maximize } \frac{1}{\|\mathbf{w}\|} \]

subject to the constraints:

\[ y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \text{ for } i = 1, 2, ..., N \]

where:
- \( N \) is the number of training examples,
- \( (\mathbf{x}_i, y_i) \) are the training examples with \( \mathbf{x}_i \) as the feature vector and \( y_i \) as the class label (-1 or 1).

**4. Soft Margin and Regularization:**

In practice, perfect separation of classes may not always be possible or desirable, especially when dealing with noisy or overlapping data. SVM allows for soft margins by introducing a regularization parameter \( C \), which controls the trade-off between maximizing the margin and minimizing the classification error.

The objective function for soft-margin SVM is:

\[ \text{minimize } \frac{1}{2} \| \mathbf{w} \|^2 + C \sum_{i=1}^{N} \xi_i \]

subject to the constraints:

\[ y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \text{ and } \xi_i \geq 0 \]

where \( \xi_i \) are slack variables that allow some data points to be misclassified or fall within the margin.

**5. Kernel Trick:**

In cases where the data is not linearly separable in the original feature space, SVM can map the input features into a higher-dimensional space using a kernel function. This allows for finding a nonlinear decision boundary in the transformed feature space.

Commonly used kernel functions include linear, polynomial, radial basis function (RBF), and sigmoid kernels.

In summary, SVM is a versatile algorithm that finds the optimal hyperplane to separate classes in the feature space by maximizing the margin. It allows for both linear and nonlinear classification using the kernel trick, making it suitable for a wide range of classification tasks.
         */
        #endregion

        #region
        /*
         Let's illustrate how Support Vector Machine (SVM) works with a simple example using Python and the widely-used Iris dataset for binary classification.

```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix

# Load the Iris dataset (using only two features for simplicity)
iris = load_iris()
X = iris.data[:, :2]  # Selecting only the first two features (sepal length and sepal width)
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot decision boundary and support vectors
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_classifier.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('Support Vector Machine Decision Boundary')
plt.show()
```

Explanation of the steps:

1. We import necessary libraries including `numpy`, `matplotlib.pyplot` for visualization, and various modules from `sklearn` for dataset loading, model creation, and evaluation.

2. We load the Iris dataset and select only the first two features (sepal length and sepal width) for simplicity.

3. The dataset is split into training and test sets using `train_test_split`.

4. We standardize the features using `StandardScaler` to ensure they have the same scale, which is important for SVM.

5. We create an SVM classifier with a linear kernel and fit it to the training data.

6. We make predictions on the test set and calculate the accuracy of the model.

7. We visualize the decision boundary and support vectors using `matplotlib.pyplot`. The decision boundary separates the two classes, and the support vectors are the data points closest to the decision boundary.

The output will display the accuracy of the SVM classifier and a plot showing the decision boundary and support vectors. This illustrates how SVM separates the classes in the feature space using a hyperplane while maximizing the margin between the classes.
         */
        #endregion
    }
}
