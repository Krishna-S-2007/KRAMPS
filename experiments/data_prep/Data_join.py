# Import libs
import pandas as pd

# Import partial datasets
data_1 = pd.read_csv('Dataset/Dataset_encoded_1.csv')
data_2 = pd.read_csv('Dataset/Dataset_encoded_2.csv')
data_3 = pd.read_csv('Dataset/Dataset_encoded_3.csv')
data_4 = pd.read_csv('Dataset/Dataset_encoded_4.csv')
data_5 = pd.read_csv('Dataset/Dataset_encoded_5.csv')
data_6 = pd.read_csv('Dataset/Dataset_encoded_6.csv')
data_7 = pd.read_csv('Dataset/Dataset_encoded_7.csv')

# Join datasets
data = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7], ignore_index=True, axis=0)

# # Check dataframe
data.info()

# # Save dataset
data.to_csv('Dataset/Dataset_final.csv', index=False)