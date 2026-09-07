# Import libs
import pandas as pd

# Import dataset
data = pd.read_csv('Dataset/Dataset_clean.csv', skiprows=6000000)

# Save preproccessed dataset
data.to_csv('Dataset/Dataset_preprocess_7.csv', index=False)