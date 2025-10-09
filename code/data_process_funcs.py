import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import sklearn
# import sklearn.preprocessing as preprocessing
# from sklearn.preprocessing import MaxAbsScaler
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
    totalErrors.rename("totalError", inplace=True)

    df2 = df2.div(totalErrors, axis=0)
    df2.fillna(0, inplace=True)
    #df2['totalError'] = totalErrors
    #df2.insert(loc=0,  value=totalErrors)
    df2 = pd.concat([pd.DataFrame(totalErrors), df2], axis=1)
    return df2

def create_unprocessed_df(file_name, shots = 4096):
    df = pd.read_csv(file_name)
    df.fillna(0, inplace=True)

    df2 = df
    # col1Name = df.iloc[0,0]

    df2.iloc[:,0] = abs(df2.iloc[:,0] - 4096)

    totalErrors = df2.iloc[:,0]
    totalErrors.rename("totalError", inplace=True)
    df2 = pd.concat([pd.DataFrame(totalErrors), df2], axis=1)
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
        df = create_processed_df(bigDf.loc[i,'file_path'])
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

def get_meta_dataframe_unprocessed(type_data, nr_qubit_sizes = 2,nr_machines = 2, nr_circuits = 3):
    bigDf =  meta_dataframe_functions.blank_meta_df()
    meta_dataframe_functions.load_meta_df(bigDf,type_data)

    df_array = []
    for i in range (nr_machines * nr_circuits * nr_qubit_sizes):
        df = create_unprocessed_df(bigDf.loc[i,'file_path'])
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

def get_expanded_df(type_data, nr_qubits, updated_results = False, updated_service = 'Default'):
    big_df =  meta_dataframe_functions.blank_meta_df()
    meta_dataframe_functions.load_meta_df(big_df,type_data, updated_results, updated_service)

    df_qubits = big_df[big_df['nr_qubits']==str(nr_qubits)]
    # df_qubits =meta_dataframe_functions.add_experiment_type_column(df_qubits)
    df_arr = []
    for i in df_qubits.index: #for each row
        df_ = create_processed_df(df_qubits.loc[i,'file_path'])
    
        default_vals = {
                'circuit_type': df_qubits.loc[i,'circuit_type'],
                'backend': df_qubits.loc[i,'backend'],
                'nr_qubits': df_qubits.loc[i,'nr_qubits'],
                'experiment_type': type_data
            }
        data = {}
        for col, val in default_vals.items():
            data[col] = [val] * len(df_)
        default_df = pd.DataFrame(data)
        df_ = pd.concat([default_df, df_], axis=1)
        df_arr.append(df_)

    df = pd.concat(df_arr)
    return df
    
def find_max_value_cols(df, num_cols = 5):
    max_values = df.iloc[:,num_cols:].max()
    return max_values

def find_min_value_cols(df, num_cols = 5):
    min_values = df.iloc[:,num_cols:].min()
    return min_values

def find_mean_cols(df, num_cols = 5):
    mean_values = df.iloc[:,num_cols:].mean()
    return mean_values

def find_std_dev_cols(df, num_cols = 5):
    std_dev_values = df.iloc[:,num_cols:].std()
    return std_dev_values

def find_variance_cols(df, num_cols = 5):
    variance_values = df.iloc[:,num_cols:].var()
    return variance_values

def find_range_dataframe(df, num_cols = 5):
    range_df = pd.DataFrame()
    max_values = find_max_value_cols(df, num_cols)
    min_values = find_min_value_cols(df, num_cols)
    mean = find_mean_cols(df, num_cols)
    std = find_std_dev_cols(df, num_cols)
    var = find_variance_cols(df, num_cols)
    range_values = max_values - min_values
    range_df = pd.concat([max_values, min_values, range_values, mean, std, var], axis=1)
    range_df.columns = ['max', 'min', 'range', 'mean', 'standard deviation', 'variance']
    return range_df