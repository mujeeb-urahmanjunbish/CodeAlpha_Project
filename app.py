import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Generate Synthetic Dataset
np.random.seed(42)
n_samples = 100

data = {
    "Temperature (\u00b0C)": np.random.normal(50, 10, n_samples).tolist(),
    "Pressure (bar)": np.random.normal(5, 1.5, n_samples).tolist(),
    "Operational Status": np.random.choice(["Operational", "Failed"], size=n_samples, p=[0.8, 0.2]),
    "Date": pd.date_range(start="2023-01-01", periods=n_samples, freq="D").tolist(),
}

# Introduce missing values and outliers
data["Temperature (\u00b0C)"][10] = None  # Missing value
data["Pressure (bar)"][20] = 15  # Outlier

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Handle Missing Values
# Fill missing values in 'Temperature (\u00b0C)' with the column mean
df["Temperature (\u00b0C)"] = df["Temperature (\u00b0C)"].fillna(df["Temperature (\u00b0C)"].mean())

# Step 3: Identify and Address Outliers
# Cap 'Pressure (bar)' values to the 1st and 99th percentiles
q1, q99 = df["Pressure (bar)"].quantile([0.01, 0.99])
df["Pressure (bar)"] = np.clip(df["Pressure (bar)"], q1, q99)

# Step 4: Normalize Features
scaler = MinMaxScaler()
df[["Temperature (\u00b0C)", "Pressure (bar)"]] = scaler.fit_transform(df[["Temperature (\u00b0C)", "Pressure (bar)"]])

# Step 5: Split the data into training and testing sets
X = df[["Temperature (\u00b0C)", "Pressure (bar)"]]
y = df["Operational Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed datasets with error handling
try:
    # Append the target variable back to the datasets
    X_train["Operational Status"] = y_train.values
    X_test["Operational Status"] = y_test.values

    # Save to CSV
    X_train.to_csv(r"D:\helloworld\preprocessed_train.csv", index=False)
    X_test.to_csv(r"D:\helloworld\preprocessed_test.csv", index=False)

    print("Training dataset saved as: D:\\helloworld\\preprocessed_train.csv")
    print("Testing dataset saved as: D:\\helloworld\\preprocessed_test.csv")
except PermissionError as e:
    print(f"Permission denied: {e}. Ensure the file is not open and you have write access.")
except Exception as e:
    print(f"An error occurred: {e}")
