import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv('california_housing_train.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Extract features and target variable
xpoints = df.drop(columns=["population"]).values
ypoints = (df["population"] > df["population"].mean()).astype(int).values  # Binarize the target variable

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(xpoints, ypoints, test_size=0.1, random_state=42)

# Create and train the Random Forest model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(x_train, y_train)

# Make predictions on the test set
ypoints_pred = random_forest.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, ypoints_pred)
print("Accuracy:", accuracy)
