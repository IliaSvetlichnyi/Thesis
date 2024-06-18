import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to categorical columns
categorical_columns = ['sex', 'smoker', 'region']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Display the updated DataFrame
print(data.head())