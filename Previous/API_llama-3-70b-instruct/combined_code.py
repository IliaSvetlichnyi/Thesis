import pandas as pd

file_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
df = pd.read_csv(file_path)

print(df.head())
print(df.info())
print(df.describe())
print(df['age'].value_counts())
print(df['sex'].value_counts())
print(df['bmi'].value_counts())
print(df['children'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())
print(df['charges'].value_counts())


import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Print the data types and summary statistics
print(df.info())
print(df.describe())

# Print the value counts for each column
for col in df.columns:
    print(df[col].value_counts())


import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Identify missing values
print(df.isnull().sum())

# Identify data types
print(df.dtypes)

# Statistical summary
print(df.describe())

# Value counts for each column
print(df['age'].value_counts())
print(df['sex'].value_counts())
print(df['bmi'].value_counts())
print(df['children'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())
print(df['charges'].value_counts())



import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Check for missing values
print(df.isnull().sum())

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[['age', 'bmi', 'children', 'charges']] = imputer.fit_transform(df[['age', 'bmi', 'children', 'charges']])

# Check again for missing values
print(df.isnull().sum())

import pandas as pd

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Convert categorical variables to numerical representations using one-hot encoding
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])

# Print the resulting dataframe
print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Split the data into training and testing sets
X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



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



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Preprocess the data
X = df.drop(['charges'], axis=1)
y = df['charges']

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and hyperparameters
models = [
    LinearRegression(),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(n_estimators=100, max_depth=5)
]

# Train and evaluate the models
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Algorithm: {model.__class__.__name__}, MSE: {mse:.2f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Preprocess the data
X = pd.get_dummies(df, columns=['sex', 'smoker', 'region']).drop(['charges'], axis=1)
y = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
mse_values = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)
    print(f"Algorithm: {model.__class__.__name__}, MSE: {mse:.2f}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Preprocess the data
X = dataset[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
y = dataset['charges']

# Encode categorical variables
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the models
models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Algorithm: {model.__class__.__name__}, MSE: {mse:.2f}')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Preprocess the data
X = dataset[['age', 'bmi', 'children', 'sex', 'smoker', 'region']]
y = dataset['charges']

# Encode categorical variables
X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the models
models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor()]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Algorithm: {model.__class__.__name__}, MSE: {mse:.2f}')

