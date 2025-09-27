import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import MaxAbsScaler

import os

def create_processed_df(file_name, shots = 4096):
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)

    df = pd.read_csv(file_name)
    df.fillna(0, inplace=True)

    df2 = df
    # col1Name = df.iloc[0,0]

    df2.iloc[:,0] = abs(df2.iloc[:,0] - 4096)

    totalErrors = df2.iloc[:,0]
    df2 = df2.div(totalErrors, axis=0)
    df2['totalError'] = totalErrors
    return df2

def create_processed_dfs(file_names_array, shots = 4096):
    dfs = []
    for file_name in file_names_array:
        df = create_processed_df(file_name, shots)
        dfs.append(df)
    return dfs