import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_sp
def dataSplit(path, split_size = 0.8):
    df = pd.read_csv(path)
  
    features = df.loc[0:6]
    print(features.shape)
    # X_train, X_test, Y_train, Y_test = train_test_split()



dataset_path = './dataset/tweets.csv'

dataSplit(dataset_path)