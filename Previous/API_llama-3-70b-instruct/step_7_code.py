

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Preprocess the data
X = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the machine learning algorithms
algorithms = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_estimators=100)
]

# Train and evaluate each algorithm
for algorithm in algorithms:
    algorithm.fit(X_train, y_train)
    y_pred = algorithm.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Algorithm: {algorithm.__class__.__name__}, MSE: {mse:.2f}")