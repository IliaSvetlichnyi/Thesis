import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = "/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv"
df = pd.read_csv(data_path)

# Preprocess the data
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# Define features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_lr = LinearRegression()

# Train the model
model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_lr = mean_squared_error(y_test, y_pred_lr)

print(f"Random Forest Regressor Score: {mse_rf}")
print(f"Linear Regression Score: {mse_lr}")