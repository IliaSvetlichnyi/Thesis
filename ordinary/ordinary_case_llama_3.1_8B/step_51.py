from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def step_51(df_encoded, data_types_info):
    X = df_encoded.drop(['charges'], axis=1)
    y = df_encoded['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def validate_step():
    df_encoded = # your dataframe here
    data_types_info = # your data types info here
    model, X_train, X_test, y_train, y_test = step_51(df_encoded, data_types_info)