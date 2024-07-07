import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_22 import step_22
from step_31 import step_31
from step_32 import step_32
from step_35 import step_35
from step_51 import step_51
from step_52 import step_52

def step_53(df):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])
    df['smoker'] = le.fit_transform(df['smoker'])
    df['region'] = le.fit_transform(df['region'])

    from sklearn.model_selection import train_test_split
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    from sklearn.metrics import mean_squared_error
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    rmse = mse ** 0.5
    return df

