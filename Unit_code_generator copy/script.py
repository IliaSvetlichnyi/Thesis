from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pandas as pd

system = {'linear_regression': LinearRegression(), 
          'decision_tree_regressor': DecisionTreeRegressor(), 
          'random_forest_regressor': RandomForestRegressor(), 
          'support_vector_regression': SVR(kernel='rbf', C=1e3) 
         }
hyperparameters = {
    'random_forest_regressor': {'n_estimators': [10, 50, 100], 'max_depth': [2, 5, 10]}
}