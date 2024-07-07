import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_22 import step_22
from step_31 import step_31
from step_32 import step_32
from step_35 import step_35
from step_51 import step_51

def step_52(df):
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred, squared=False)
    
    if mse > 5000:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, squared=False)
        
        if mse > 5000:
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred, squared=False)
    
    df['charges'] = model.predict(df.drop('charges', axis=1))
    
    return df