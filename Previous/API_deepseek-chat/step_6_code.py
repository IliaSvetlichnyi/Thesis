import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data_path = '/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv'
df = pd.read_csv(data_path)

# Define the features and target
features = df.drop('charges', axis=1)
target = df['charges']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)