import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/Iris.csv')

imputer = SimpleImputer(strategy='mean')
df[['sepal_length', 'sepal_width']] = imputer.fit_transform(df[['sepal_length', 'sepal_width']])

categorical_cols = ['species']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', pd.get_dummies)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', df.select_dtypes(include=['int64', 'float64'])),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X = preprocessor.fit_transform(df.drop('species', axis=1))
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))