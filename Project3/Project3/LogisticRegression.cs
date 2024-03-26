using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project3
{
    internal class LogisticRegression
    {
        #region exp
        /*
         Logistic regression is a statistical model used for binary classification tasks, where the outcome variable (dependent variable) is categorical and has only two possible classes. Despite its name, logistic regression is primarily used for classification rather than regression. In logistic regression, the predicted output is a probability score that the given input belongs to a particular class. It's widely employed in various fields such as medicine, finance, and social sciences. Let's delve deeper into the explanation of logistic regression:

**1. Sigmoid Function:**

In logistic regression, the logistic or sigmoid function is used to model the relationship between the independent variables and the probability of the binary outcome. The sigmoid function is defined as:

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

where \( z \) is the linear combination of the input features and model coefficients.

**2. Logistic Regression Model:**

The logistic regression model predicts the probability of an observation belonging to the positive class (usually labeled as 1) as follows:

\[ P(y=1 \,|\, \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) \]

where:
- \( P(y=1 \,|\, \mathbf{x}) \) is the probability that \( y \) (the outcome) is 1 given the input features \( \mathbf{x} \),
- \( \sigma(\cdot) \) is the sigmoid function,
- \( \mathbf{w} \) is the vector of coefficients (weights),
- \( \mathbf{x} \) is the vector of input features,
- \( b \) is the bias term (intercept).

**3. Decision Boundary:**

The decision boundary is the threshold value (usually 0.5) that separates the two classes. If the predicted probability is above the threshold, the observation is classified as belonging to the positive class (1); otherwise, it's classified as belonging to the negative class (0).

**4. Training Logistic Regression:**

The logistic regression model is trained using a method called maximum likelihood estimation (MLE), where the parameters (coefficients) are adjusted to maximize the likelihood of observing the given data under the assumed model.

**5. Regularization:**

To prevent overfitting, regularization techniques like L1 (Lasso) and L2 (Ridge) regularization can be applied to penalize large coefficient values.

**6. Evaluation Metrics:**

Common evaluation metrics for logistic regression include accuracy, precision, recall, F1 score, and ROC-AUC (Receiver Operating Characteristic - Area Under Curve).

**7. Applications of Logistic Regression:**

- Spam detection in emails.
- Customer churn prediction in telecommunications.
- Credit risk assessment in finance.
- Disease prediction in healthcare.
- Sentiment analysis in natural language processing.

**8. Interpretability:**

Logistic regression models are relatively interpretable, as the coefficients indicate the direction and strength of the relationship between the input features and the outcome. Positive coefficients indicate a positive relationship with the log-odds of the outcome, while negative coefficients indicate a negative relationship.

In summary, logistic regression is a powerful and interpretable model for binary classification tasks, commonly used in various domains for making predictions and understanding relationships between variables.
         */
        #endregion exp

        #region math
        /*
         In logistic regression, the logistic or sigmoid function is used to model the relationship between the independent variables and the probability of the binary outcome. The logistic function is defined as:

\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]

where \( z \) is the linear combination of the input features and model coefficients:

\[ z = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n \]

where:
- \( w_0 \) is the bias term (intercept),
- \( w_1, w_2, \ldots, w_n \) are the coefficients (weights) corresponding to the input features \( x_1, x_2, \ldots, x_n \).

The logistic regression model predicts the probability \( P(y=1 \,|\, \mathbf{x}) \) that the outcome \( y \) is 1 (positive class) given the input features \( \mathbf{x} \) using the logistic function:

\[ P(y=1 \,|\, \mathbf{x}) = \sigma(z) \]

Substituting \( z \) into the sigmoid function, we get:

\[ P(y=1 \,|\, \mathbf{x}) = \frac{1}{1 + e^{-(w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n)}} \]

Similarly, the probability \( P(y=0 \,|\, \mathbf{x}) \) that the outcome \( y \) is 0 (negative class) given the input features \( \mathbf{x} \) can be expressed as:

\[ P(y=0 \,|\, \mathbf{x}) = 1 - P(y=1 \,|\, \mathbf{x}) \]

The decision boundary separates the two classes. If the predicted probability \( P(y=1 \,|\, \mathbf{x}) \) is greater than or equal to 0.5, the observation is classified as belonging to the positive class (1); otherwise, it's classified as belonging to the negative class (0).

During training, the model's parameters (coefficients \( w_0, w_1, \ldots, w_n \)) are estimated using a method called maximum likelihood estimation (MLE). The goal is to find the values of \( w_0, w_1, \ldots, w_n \) that maximize the likelihood of observing the given data under the assumed logistic regression model.

Regularization techniques such as L1 (Lasso) and L2 (Ridge) regularization can be applied to logistic regression to prevent overfitting by penalizing large coefficient values.

In summary, logistic regression models the relationship between input features and the probability of a binary outcome using the logistic function, providing a powerful tool for binary classification tasks.
         */
        #endregion math

        #region example
        /*
         Let's provide an example of logistic regression using a binary classification problem. Suppose we have a dataset containing information about whether students passed (1) or failed (0) an exam based on the number of hours they studied. We want to build a logistic regression model to predict whether a student will pass the exam based on the number of hours they studied.

Here's how we can implement logistic regression using Python's `scikit-learn` library:

```python
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example data
hours_studied = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Independent variable
exam_results = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])  # Dependent variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(hours_studied, exam_results, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

In this example:
- We first import necessary libraries including `numpy`, `train_test_split` from `sklearn.model_selection`, `LogisticRegression` from `sklearn.linear_model`, and `accuracy_score` from `sklearn.metrics`.
- We create sample data where `hours_studied` represents the number of hours students studied, and `exam_results` represents whether they passed (1) or failed (0) the exam.
- We split the data into training and test sets using `train_test_split`.
- We create a logistic regression model using `LogisticRegression` and fit it to the training data using `fit`.
- We make predictions on the test set using `predict`.
- Finally, we calculate the accuracy of the model using `accuracy_score` by comparing the true labels (`y_test`) with the predicted labels (`y_pred`).

The output will display the accuracy of the logistic regression model on the test set. This accuracy represents the proportion of correctly classified instances.
         */
        #endregion example
    }
}
