import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_22 import step_22
from step_31 import step_31
from step_32 import step_32
from step_35 import step_35
from step_51 import step_51
from step_52 import step_52
from step_53 import step_53

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def step_61(df):
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    df_performance = pd.DataFrame({'Metric': ['Mean Squared Error', 'R-Squared'], 'Value': [mse, r2]})
    return df_performance