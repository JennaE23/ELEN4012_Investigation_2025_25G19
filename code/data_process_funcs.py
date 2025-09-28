import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import MaxAbsScaler
import meta_dataframe_functions

import os

def create_processed_df(file_name, shots = 4096):
    # current_directory = os.getcwd()
    # print("Current working directory:", current_directory)

    df = pd.read_csv(file_name)
    df.fillna(0, inplace=True)

    df2 = df
    # col1Name = df.iloc[0,0]

    df2.iloc[:,0] = abs(df2.iloc[:,0] - 4096)

    totalErrors = df2.iloc[:,0]
    df2 = df2.div(totalErrors, axis=0)
    # df2['totalError'] = totalErrors
    # df2.insert(loc=0, column='totalError', value=totalErrors)
    df2 = pd.concat([pd.DataFrame(totalErrors, columns=['totalError']), df2], axis=1)
    return df2

def create_processed_dfs(file_names_array, shots = 4096):
    dfs = []
    for file_name in file_names_array:
        df = create_processed_df(file_name, shots)
        dfs.append(df)
    return dfs

def get_meta_dataframe(type_data, nr_qubit_sizes = 2,nr_machines = 2, nr_circuits = 3):
    bigDf =  meta_dataframe_functions.blank_meta_df()
    meta_dataframe_functions.load_meta_df(bigDf,type_data)

    df_array = []
    for i in range (nr_machines * nr_circuits * nr_qubit_sizes):
        df = create_processed_df(bigDf['file_path'][i])
        # df.insert(loc=0, column='circuit_type', value=bigDf['circuit_type'][i])
        # df.insert(loc=0, column='backend', value=bigDf['backend'][i])
        # df.insert(loc=0, column='nr_qubits', value=bigDf['nr_qubits'][i])
        default_vals = {
            'circuit_type': bigDf['circuit_type'][i],
            'backend': bigDf['backend'][i],
            'nr_qubits': bigDf['nr_qubits'][i]
        }
        data = {}
        for col, val in default_vals.items():
            data[col] = [val] * len(df)
        default_df = pd.DataFrame(data)
        df = pd.concat([default_df, df], axis=1)

        df_array.append(df)
    return df_array

def join_dfs(dfs):
    bigDf = pd.DataFrame()
    for df in dfs:
        bigDf = pd.concat([bigDf, df], ignore_index=True)
    return bigDf

def arr_dfs_of_qubit_sizes(type_data, nr_qubit_sizes = 2,nr_machines = 2, nr_circuits = 3):
    df_arr = get_meta_dataframe(type_data, nr_qubit_sizes, nr_machines, nr_circuits)
    df = []
    for i in range (nr_qubit_sizes):
        start_index = i * nr_machines * nr_circuits
        end_index = (i + 1) * nr_machines * nr_circuits
        df_size = join_dfs(df_arr[start_index:end_index])
        df.append(df_size)
    return df
    