import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import PolynomialFeatures
import pickle

# Load the dataset from the CSV file
df = pd.read_csv("ml_dataset.csv")


# Separate features (X) and target (y)
X = df.drop("HR", axis=1)  # Features without the "HR" column
y = df["HR"]  # Target is the "HR" column

# Split the data into training and validation sets
X_train, X_test, y_train, y_valid = train_test_split(X, y, test_size=0.24, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('standard_scaler', StandardScaler()),  # Step 1: StandardScaler
    ('pca', PCA(n_components=22)),  # Step 2: PCA (Principal Component Analysis)
    ("Polynomial", PolynomialFeatures(degree=3)),  # Step 3: PolynomialFeatures
    ("LinearRegression", LinearRegression())  # Step 4: Linear Regression
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

#Continue with training and predictions using pipeline
predictions = pipeline.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_valid, predictions)

# Calculate root mean squared error
rmse = np.sqrt(mse)

# Calculate R-squared score
r2 = r2_score(y_valid, predictions)

# Print the evaluation metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Save the trained pipeline to a file named 'model.pkl'
pickle.dump(pipeline, open('model.pkl', 'wb'))


