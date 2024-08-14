

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def step_51(df_encoded):
    X = df_encoded.drop(['charges'], axis=1)
    y = df_encoded['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', max_depth=5, learning_rate=0.1, n_estimators=1000)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test