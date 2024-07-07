import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data_path = "/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv"
df = pd.read_csv(data_path)

# Preprocess the data
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['region'] = df['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

# Split the data into training and testing sets
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
lr_model = LinearRegression()

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Evaluate the models on the testing data
rf_score = rf_model.score(X_test, y_test)
lr_score = lr_model.score(X_test, y_test)

rf_rmse = mean_squared_error(y_test, rf_model.predict(X_test), squared=False)
lr_rmse = mean_squared_error(y_test, lr_model.predict(X_test), squared=False)

print(f"Random Forest Regressor Score: {rf_score}")
print(f"Linear Regression Score: {lr_score}")

print(f"Random Forest Regressor RMSE: {rf_rmse}")
print(f"Linear Regression RMSE: {lr_rmse}")