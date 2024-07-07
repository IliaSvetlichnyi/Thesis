from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split('/Users/ilya/Desktop/GitHub_Repositories/HW_University/Data_Mining/datasets/insurance.csv', 
                                        test_size=0.2, 
                                        random_state=42)