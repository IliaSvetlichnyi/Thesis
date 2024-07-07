import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_22 import step_22
from step_31 import step_31
from step_32 import step_32
from step_35 import step_35

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def step_51(df):
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
    
    return df