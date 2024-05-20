import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Perceptron model
perceptron = Perceptron(random_state=42)
perceptron.fit(X_train, y_train)

# Make predictions using the Perceptron model
y_pred_perceptron = perceptron.predict(X_test)

# Calculate accuracy of the Perceptron model
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)

# Create and train the Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=200)
log_reg.fit(X_train, y_train)

# Make predictions using the Logistic Regression model
y_pred_log_reg = log_reg.predict(X_test)

# Calculate accuracy of the Logistic Regression model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# Print the accuracies
print("Accuracy of Perceptron: {:.2f}%".format(accuracy_perceptron * 100))
print("Accuracy of Logistic Regression: {:.2f}%".format(accuracy_log_reg * 100))
