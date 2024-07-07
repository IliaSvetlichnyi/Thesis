import pandas as pd
from step_11 import step_11
from step_21 import step_21
from step_22 import step_22
from step_31 import step_31
from step_32 import step_32

import pandas as pd
from sklearn.model_selection import train_test_split

def step_35(df: pd.DataFrame) -> tuple:
    X = df.drop('charges', axis=1)
    y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test