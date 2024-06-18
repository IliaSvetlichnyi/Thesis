import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score

data = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

if 'target' in data.columns:
    if len(data.select_dtypes(include=['int64']).columns) > 1:
        alg = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        alg = LogisticRegression(random_state=42)
else:
    alg = LinearRegression()

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

alg.fit(X_train, y_train)

y_pred = alg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'Accuracy: {accuracy}')

if data['target'].dtype == 'int64':
    print('Target variable is categorical. Using decision trees.')
    alg = DecisionTreeClassifier(random_state=42) if len(data.select_dtypes(include=['object']).columns) > 0 else DecisionTreeRegressor()
else:
    print('Target variable is continuous. Using linear regression.')

X_train, X_test, y_train, y_test = model_selection.train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

alg.fit(X_train, y_train)

y_pred = alg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) if data['target'].dtype == 'int64' else r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'Accuracy: {accuracy}')