import itertools
from itertools import combinations
import pandas as pd
import numpy as np

from investigation_functions import data_process_funcs as dpf
from investigation_functions import ml_funcs as mlf
#/////////////////////////////////////////////////
#comparison test things
def split_into_circuits(df_all_circuits):
    circuits = df_all_circuits.groupby('circuit_type')
    circuit_1 = circuits.get_group(1)
    circuit_2 = circuits.get_group(2)
    circuit_3 = circuits.get_group(3)
    return [circuit_1,circuit_2,circuit_3]

def make_same_backends(dfs,backends):
    dfs_ = dfs.copy()
    dfs_mod = []
    for df in dfs_:
        df = df[df['backend'].isin(backends)]
        dfs_mod.append(df)
    
    return dfs_mod

def get_HSR_array_all_backends(nr_qubits,dir_='', updated_results = False):

    df_H = dpf.get_expanded_df('Hardware',nr_qubits, dir_,updated_results)
    df_S = dpf.get_expanded_df('Simulation',nr_qubits,dir_, updated_results)
    df_R = dpf.get_expanded_df('Refreshed_Simulation',nr_qubits,dir_, updated_results)

    return [df_H, df_S, df_R]

def get_circuit_type_array(df_nq):
    df_nqi = mlf.features_to_int(df_nq)
    circuits = split_into_circuits(df_nqi) 
    return circuits

def get_HSR_test_table(initial_list):
    
    list_of_arrays = generate_combos(initial_list)
    
    df_SR = pd.concat(initial_list[1:3].copy())
    df_HS = pd.concat(initial_list[0:2].copy())
    df_HR = pd.concat([initial_list[0],initial_list[2]])
    df_HSR = pd.concat([df_HS,initial_list[2]])
    # Quick and dirty fix for results_to_csv function:
    df_SR['experiment_type'] = 'Sim and Refreshed'
    df_HS['experiment_type'] = 'Hardware and Sim'
    df_HR['experiment_type'] = 'Hardware and Refreshed'
    df_HSR['experiment_type'] = 'Hardware, Sim, and Refreshed'
    #Train on H, Test on SR combined:
    list_of_arrays[0].append(df_SR)
    list_of_arrays.append([df_HS,initial_list[2]])
    list_of_arrays.append([df_HR,initial_list[1]])
    list_of_arrays.append([df_SR,initial_list[0]])
    list_of_arrays.append([df_HSR,df_HSR])
    #make the train on H row only torino and brisbane:
    #H_backends = initial_list[0]['backend'].unique()
    for i in range(len(initial_list)):
        backends = initial_list[i]['backend'].unique()
        list_of_arrays[i] = make_same_backends(list_of_arrays[i],backends)
    
    #Train on SR and Test on H only:
    # train_SR_test_H = [df_SR,initial_list[0]]
    # list_of_arrays.append(train_SR_test_H)

    return list_of_arrays

def get_circuits_test_table(df_nq):
    circuits = get_circuit_type_array(df_nq)
    combos = generate_combos(circuits,True)

    # add the combined training rows
    for i in range(len(circuits)):
        combined_train = [combos[i][3],combos[i][0]]
        combos.append(combined_train)
    
    return combos

def generate_combos(individual_dfps,include_combined=False):
    indiv_list = individual_dfps.copy()
    nr_indiv = len(indiv_list)
    table =[]

    row1 = indiv_list.copy()
    if include_combined:
        #make elements joined as pairs
        pair_dfs = make_pairs(row1[1:3])
        #append the paired elements
        row1 = row1 +pair_dfs
    table.append(row1)

    for i in range(1,nr_indiv):
        row = indiv_list.copy()
        train_element =  row.pop(i)
        row.insert(0,train_element)
        
        if include_combined:
            pair_parts = row[1:3].copy()
            #make elements joined as pairs
            pair_dfs = make_pairs(pair_parts)
            #append the paired elements
            row = row +pair_dfs

        table.append(row)    

    return table

def make_pairs(indiv_dfs):
    pairs = list(combinations(indiv_dfs.copy(), 2))
    pair_dfs = []
    for pair in pairs:
        df = pd.concat(pair[:])
        pair_dfs.append(df)

    return pair_dfs

def print_test_table(test_table,Exp_type = True,backends = True, circ_types = True, nans = False):

    for row in test_table:
        print("\nrow")
        for df in row:
            print('\ndf')
            if Exp_type:
                print(df['experiment_type'].unique())
            if backends:
                print(df['backend'].unique())
            if circ_types:
                print(df['circuit_type'].unique())
            if nans:
                print(df.isna().sum())