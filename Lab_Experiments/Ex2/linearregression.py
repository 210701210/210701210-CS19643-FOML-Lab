import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Load the data
df = pd.read_csv('california_housing_train.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Extract features and target variable
xpoints = df["longitude"].values.reshape(-1, 1)
ypoints = df["population"].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(xpoints, ypoints, test_size=0.1, random_state=42)

# Create and train the linear regression model
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

# Make predictions on the test set
ypoints_pred = reg.predict(x_test)

# Plot the results
plt.scatter(x_test, y_test, color="red", label="Actual")
plt.plot(x_test, ypoints_pred, color="blue", label="Predicted")
plt.xlabel("Longitude")
plt.ylabel("Population")
plt.title("Linear Regression: Longitude vs Population")
plt.legend()
plt.show()
