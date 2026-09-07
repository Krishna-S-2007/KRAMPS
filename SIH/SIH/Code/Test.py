# Import libs
import pandas as pd

# Import dataset
data = pd.read_csv('Dataset/Dataset_final.csv')

data.info()

print(data.isna().sum())
print(data.columns)