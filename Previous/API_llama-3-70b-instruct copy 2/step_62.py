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
from step_61 import step_61



import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def step_62(df):
    y_true = df['charges']
    y_pred = [...]  # Replace with your predicted values
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    df['accuracy'] = [accuracy] * len(df)
    df['precision'] = [precision] * len(df)
    df['recall'] = [recall] * len(df)
    df['f1-score'] = [f1] * len(df)
    
    return df