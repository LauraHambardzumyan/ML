using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project3
{
    internal class NaiveBayes
    {

        #region
        /*
         Naive Bayes is a popular and simple machine learning algorithm based on Bayes' theorem. It's particularly useful for classification tasks and is widely used in various applications such as text classification, spam filtering, medical diagnosis, and sentiment analysis. Despite its simplicity, Naive Bayes often performs surprisingly well, especially on datasets with high dimensionality.

**1. Meaning and Underlying Principle:**

Naive Bayes is based on Bayes' theorem, which describes the probability of an event based on prior knowledge of conditions that might be related to the event. In the context of classification, Naive Bayes calculates the probability of a particular class label given the input features.

**2. Assumption of Conditional Independence:**

The "naive" in Naive Bayes refers to the assumption of conditional independence among the features given the class label. This means that the presence of one feature is assumed to be independent of the presence of any other feature, which is often not true in practice. Despite this simplifying assumption, Naive Bayes can still perform well in many real-world scenarios, especially when the features are approximately independent or when the model is robust to violations of this assumption.

**3. Usage:**

- **Text Classification:** Naive Bayes is commonly used for text classification tasks such as spam detection, sentiment analysis, and topic classification. In these applications, the input features are typically word frequencies or presence/absence of certain words in a document.

- **Medical Diagnosis:** Naive Bayes can be used in medical diagnosis to predict the likelihood of a patient having a particular disease based on symptoms and medical test results.

- **Recommendation Systems:** Naive Bayes can be used in recommendation systems to predict user preferences or to classify items into different categories based on user behavior.

- **Customer Segmentation:** Naive Bayes can be used in marketing to segment customers into different groups based on their demographics, purchase history, and other characteristics.

**4. Advantages:**

- **Simple and Fast:** Naive Bayes is computationally efficient and easy to implement, making it suitable for large datasets.
- **Requires Small Amount of Training Data:** Naive Bayes can perform well even with a small amount of training data.
- **Handles High Dimensionality:** Naive Bayes performs well on datasets with a large number of features, making it suitable for text classification and other high-dimensional data.

**5. Limitations:**

- **Assumption of Conditional Independence:** The assumption of conditional independence may not hold true in many real-world scenarios, which can lead to suboptimal performance.
- **Sensitivity to Feature Correlation:** Naive Bayes can perform poorly if there are strong correlations between features.
- **Difficulty Handling Continuous Features:** Naive Bayes assumes that features are categorical or discrete, so it may not perform well with continuous features without additional preprocessing.

In summary, Naive Bayes is a simple yet powerful algorithm for classification tasks, especially in scenarios with high dimensionality and discrete features. While its simplicity and efficiency make it attractive, practitioners should be aware of its assumptions and limitations when applying it to real-world problems.
         */
        #endregion

        #region
        /*
         The mathematics behind Naive Bayes involves applying Bayes' theorem along with the assumption of conditional independence among the features given the class label. Let's break down the math for the two main types of Naive Bayes classifiers: Gaussian Naive Bayes and Multinomial Naive Bayes.

**1. Gaussian Naive Bayes:**

Gaussian Naive Bayes is used when the features are continuous and assumed to follow a Gaussian (normal) distribution.

**Bayes' Theorem:**

\[ P(y | x_1, x_2, ..., x_n) = \frac{P(y) \cdot P(x_1, x_2, ..., x_n | y)}{P(x_1, x_2, ..., x_n)} \]

**Assumption of Conditional Independence:**

\[ P(x_1, x_2, ..., x_n | y) = P(x_1 | y) \cdot P(x_2 | y) \cdot ... \cdot P(x_n | y) \]

**Gaussian Probability Density Function:**

\[ P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \cdot \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right) \]

Where:
- \( P(y | x_1, x_2, ..., x_n) \) is the posterior probability of class \( y \) given the features \( x_1, x_2, ..., x_n \),
- \( P(y) \) is the prior probability of class \( y \),
- \( P(x_1, x_2, ..., x_n | y) \) is the likelihood of observing the features given class \( y \),
- \( P(x_1, x_2, ..., x_n) \) is the total probability of observing the features,
- \( \mu_{y,i} \) is the mean of feature \( x_i \) for class \( y \),
- \( \sigma_{y,i}^2 \) is the variance of feature \( x_i \) for class \( y \).

**2. Multinomial Naive Bayes:**

Multinomial Naive Bayes is used when the features are categorical or discrete, such as word counts in text classification.

**Bayes' Theorem:**

\[ P(y | x_1, x_2, ..., x_n) = \frac{P(y) \cdot P(x_1, x_2, ..., x_n | y)}{P(x_1, x_2, ..., x_n)} \]

**Assumption of Conditional Independence:**

\[ P(x_1, x_2, ..., x_n | y) = P(x_1 | y) \cdot P(x_2 | y) \cdot ... \cdot P(x_n | y) \]

**Multinomial Probability Distribution:**

\[ P(x_i | y) = \frac{N_{yi} + \alpha}{N_y + \alpha \cdot n} \]

Where:
- \( N_{yi} \) is the count of feature \( x_i \) for class \( y \),
- \( N_y \) is the total count of all features for class \( y \),
- \( \alpha \) is the Laplace smoothing parameter (a smoothing technique to handle unseen features),
- \( n \) is the total number of possible features.

In both cases, to classify a new instance, we calculate the posterior probability of each class given the features and choose the class with the highest probability.

This is a simplified overview of the mathematics behind Naive Bayes. The exact implementation may vary depending on the specific variant of Naive Bayes and any additional assumptions or optimizations applied.
         */
        #endregion

        #region
        /*
         Let's demonstrate how to implement Gaussian Naive Bayes and Multinomial Naive Bayes classifiers using Python's scikit-learn library. We'll use the famous Iris dataset for this example, which contains measurements of iris flowers and their corresponding species.

**1. Gaussian Naive Bayes Example:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Gaussian Naive Bayes Accuracy:", accuracy)
```

**2. Multinomial Naive Bayes Example:**

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train Multinomial Naive Bayes classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = mnb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Multinomial Naive Bayes Accuracy:", accuracy)
```

In both examples:

- We first import necessary libraries including `load_iris` to load the Iris dataset, `train_test_split` to split the data into training and test sets, `GaussianNB` and `MultinomialNB` from `sklearn.naive_bayes` to create the Gaussian Naive Bayes and Multinomial Naive Bayes classifiers respectively, and `accuracy_score` from `sklearn.metrics` to calculate accuracy.

- We load the Iris dataset containing features (sepal length, sepal width, petal length, petal width) and target (species).

- The dataset is split into training and test sets using `train_test_split`.

- We create and train the Naive Bayes classifiers using `fit` method with training data.

- We make predictions on the test set using `predict` method.

- Finally, we calculate the accuracy of the models using `accuracy_score` by comparing the predicted labels with the true labels.
         */
        #endregion
    }
}
