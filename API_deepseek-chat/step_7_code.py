import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
df = pd.read_csv(data_path)

# Preprocessing
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])

# Define features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose appropriate machine learning algorithms
# Since the target variable 'charges' is a continuous variable, we use regression algorithms

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the models
rf_score = rf.score(X_test, y_test)
lr_score = lr.score(X_test, y_test)

print(f"Random Forest Regressor Score: {rf_score}")
print(f"Linear Regression Score: {lr_score}")