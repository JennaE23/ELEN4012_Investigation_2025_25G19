import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import sklearn
from sklearn import model_selection, svm
from sklearn.svm import SVC
# from sklearn.preprocessing import LabelEncoder
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import data_process_funcs as dpf
#import meta_dataframe_functions as mdf

from sklearn.model_selection import cross_val_score

from csv import DictWriter
import itertools
from itertools import combinations


def features_to_int(df):
    df_ = df
    df_['circuit_type'] = pd.to_numeric(df_['circuit_type'], downcast='integer', errors='coerce')
    df_['nr_qubits'] = pd.to_numeric(df_['nr_qubits'], downcast='integer', errors='coerce')
    return df_

def label_encode_backend(df):
    df_ = df
    labels = {'brisbane':0, 'fez':1, 'marrakesh':2,'torino':3}# alphabetical order
    df['backend'] = df['backend'].map(labels)
    return df_

def drop_0th_col(nr_qubits,df):
    df_ = df
    col_name = np.strings.multiply('0',nr_qubits)[0]
    df_ =df_.drop(col_name,axis=1)
    return df_ 

def total_Err_to_percent(df): #in place of scaling
    df_ = df
    df_['totalError']= df_['totalError'].div(4096)
    return df_

def apply_custom_scaling(df):
    df_ = df
    df_['circuit_type'] = df_['circuit_type'].div(3)
    return df_

def apply_preprosessing(df, drop_exp_type = True,label_encode = True):#assumes only 1 nr of qubits
    df_ = df
    if drop_exp_type:
        df_ = df_.drop('experiment_type',axis = 1)
    if label_encode:
        df_ = label_encode_backend(df_)
    df_ = features_to_int(df_)
    df_ = drop_0th_col(df_[['nr_qubits']].iloc[0],df_)
    df_ = df_.drop('nr_qubits', axis = 1)
    df_ = total_Err_to_percent(df_)
    df_ = apply_custom_scaling(df_)
    return df_

def get_x_y(df_q):
    Y = df_q[['backend']]
    X = df_q.drop('backend',axis = 1)
    return X,Y

def fit_and_get_score(model,X_train,Y_train,X_test,Y_test,ravel = True, to_print = False):
    model_ = model
    if ravel:
        Y_train_1d = Y_train.to_numpy()
        Y_train_1d = Y_train_1d.ravel()
    else:
        Y_train_1d = Y_train
    #print(Y_train_1d)
    model_.fit(X_train, Y_train_1d)
    Cscore = model_.score(X_test, Y_test)
    if to_print:
        print("Accuracy:", Cscore)

    return model_,Cscore

def get_cv_score(model, X_train,Y_train,folds = 5,ravel=True, to_print = False):
    model_ = model
    if ravel:
        Y_train_1d = Y_train.to_numpy()
        Y_train_1d = Y_train_1d.ravel()
    else:
        Y_train_1d = Y_train
    score = cross_val_score(model_, X_train, Y_train_1d, cv=folds, scoring='accuracy')
    if to_print:
        print("Cross-validation accuracy: ",score)
    return score

def std_split_fit_and_scores(dfp,model, test_size_ = 0.2,fold_ = 5,cv = True):
    X,Y = get_x_y(dfp)

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X,Y,test_size=test_size_,shuffle = True,random_state=42)

    fitted_model, score = fit_and_get_score(model,X_train,Y_train,X_test,Y_test)
    cv_score = np.nan
    if cv:
        cv_score = get_cv_score(fitted_model,X_train,Y_train,folds = fold_)
    return fitted_model, score, cv_score

#/////////////////////////////////////////////////////////////
#csv things
def create_ml_results_csv(ml_alg, dir = '../ML_Results/'):
    general_fields = ['nr_qubits','machines','tr&v exp_type','tr&v circuits', 'test exp_type','test circuits','preprocess settings']
    score_fields = ['accuracy','cv_1','cv_2','cv_3','cv_4','cv_5']
    if ml_alg == 'SVM':
        file_name = dir + 'SVM_results.csv'
        # ml_param_fields = ['C', 'break_ties', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose']
        # SVM example parameters = {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1, 'probability': False, 'random_state': None, 'shrinking': True, 'tol': 0.001, 'verbose': False}
        ml_param_fields = ['kernal', 'param settings']
    else:
        file_name = dir + 'KNN_results.csv'
        # ml_param_fields = ['algorithm', 'leaf_size', 'metric', 'metric_params', 'n_jobs', 'n_neighbors', 'p', 'weights']
        # KNN example parameters = {'algorithm': 'auto',
            #  'leaf_size': 30,
            #  'metric': 'minkowski',
            #  'metric_params': None,
            #  'n_jobs': None,
            #  'n_neighbors': 5,
            #  'p': 2,
            #  'weights': 'uniform'}
        ml_param_fields = ['n_neighbors', 'param settings']
    fields = general_fields + ml_param_fields + score_fields
    with open(file_name, 'w', newline='') as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writeheader()
    return file_name, fields

def get_general_fields(nr_qubits, machines, tr_val_exp_type, tr_val_circuits, test_exp_type, test_circuits, preprocess_settings):
    general_fields = {
        'nr_qubits': nr_qubits,
        'machines': machines,
        'tr&v exp_type': tr_val_exp_type,
        'tr&v circuits': tr_val_circuits,
        'test exp_type': test_exp_type,
        'test circuits': test_circuits,
        'preprocess settings': preprocess_settings
    }
    return general_fields

def get_ml_fields(ml_alg, model_parameters, param_settings = 0):
    if ml_alg == 'SVM':
        ml_fields = {
            'kernal': model_parameters.get('kernel', 'N/A'),
            # 'param settings': str(model_parameters)
            'param settings': param_settings
        }
    else:
        ml_fields = {
            'n_neighbors': model_parameters.get('n_neighbors', 'N/A'),
            # 'param settings': str(model_parameters)
            'param settings': param_settings
        }
    return ml_fields

def get_results_fields(score, cv_score = [np.nan,np.nan,np.nan,np.nan,np.nan]):
    results_fields = {
        'accuracy': score,
        'cv_1': cv_score[0],
        'cv_2': cv_score[1],
        'cv_3': cv_score[2],
        'cv_4': cv_score[3],
        'cv_5': cv_score[4]
    }
    return results_fields

def ml_results_to_csv(general_fields, ml_fields, results_fields,file_name,fields):
    all_fields = {**general_fields, **ml_fields, **results_fields}
    with open(file_name, 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writerow(all_fields)

def get_machine_binary(machine_list, all_machines = ['torino', 'brisbane', 'fez', 'marakesh']):
    binary_list = [1 if machine in machine_list else 0 for machine in all_machines]
    binary = "".join(str(x) for x in binary_list)
    return binary

def get_machine_binary_from_df(df, all_machines = ['torino', 'brisbane', 'fez', 'marakesh']):
    machine_list = df['backend'].unique().tolist()
    binary_list = [1 if machine in machine_list else 0 for machine in all_machines]
    binary = "".join(str(x) for x in binary_list)
    return binary

def get_circuit_binary(circuit_list, all_circuits = ['1','2','3']):
    binary_list = [1 if circuit in circuit_list else 0 for circuit in all_circuits]
    binary = "".join(str(x) for x in binary_list)
    return binary

def get_circuit_binary_from_df(df, all_circuits = ['1','2','3']):
    circuit_list = df['circuit_type'].unique().astype(str).tolist()
    # print(circuit_list)
    circuit_list = [str(i) for i in circuit_list]  # Ensure all elements are strings
    binary_list = [1 if circuit in circuit_list else 0 for circuit in all_circuits]
    binary = "".join(str(x) for x in binary_list)
    return binary

#/////////////////////////////////////////////////////////////
# Big Function + baby funcs
def preprocess_dfs(dfs, preprocessing_settings):
    # Example preprocessing based on settings
    # print("Preprocessing settings:", preprocessing_settings)
    processed_dfs = []
    for df in dfs:
        match preprocessing_settings:
            case 0:
                df_processed = apply_preprosessing(df) 
            case _:
                raise ValueError("Unsupported preprocessing setting")

        processed_dfs.append(df_processed)
        # Add more preprocessing options as needed
    return processed_dfs

def KNN_model_setup(base_parameter, param_settings):
    match param_settings:
        case 0:
            model = KNeighborsClassifier(n_neighbors=base_parameter)
        case 1:
            model = KNeighborsClassifier(n_neighbors=base_parameter, weights="distance", p=1)
        case _:
            raise ValueError("Unsupported parameter setting for KNN")
    return model

def SVM_model_setup(base_parameter, param_settings):
    match param_settings:
        case 0:
            model = svm.SVC(kernel=base_parameter)
        case _:
            raise ValueError("Unsupported parameter setting for SVM")
    return model

def get_file_name_and_fields(ml_algorithm, dir = '../ML_Results/'):
    general_fields = ['nr_qubits','machines','tr&v exp_type','tr&v circuits', 'test exp_type','test circuits','preprocess settings']
    score_fields = ['accuracy','cv_1','cv_2','cv_3','cv_4','cv_5']
    match ml_algorithm:
        case 'SVM':
            file_name = dir + 'SVM_results.csv'
            ml_param_fields = ['kernal', 'param settings']
        case 'KNN':
            file_name = dir + 'KNN_results.csv'
            ml_param_fields = ['n_neighbors', 'param settings']
        case _:
            raise ValueError("Unsupported ML algorithm")
    fields = general_fields + ml_param_fields + score_fields
    return file_name, fields

def run_and_print_ml_results(train_df,test_dfs,ml_algorithm,base_parameter, dir = '../ML_Results/', get_self_score = True, preprocessing_settings = 0, param_settings = 0, cross_validation = False):
    # Get CSV setup
    nr_qubits = train_df['nr_qubits'].iloc[0]
    machines = get_machine_binary_from_df(train_df)
    tr_val_circuits = get_circuit_binary_from_df(train_df)
    tr_val_exp_type = train_df['experiment_type'].iloc[0]
    filename, fields = get_file_name_and_fields(ml_algorithm, dir)
    # if len(test_dfs) != 0:
    #     test_exp_type = test_dfs[0]['experiment_type'].iloc[0]      #Assumes all test dfs are of the same type
    
    # # Apply preprocessing
    # processed_dfs = preprocess_dfs([train_df] + test_dfs, preprocessing_settings)
    # train_df_processed = processed_dfs[0]
    # test_dfs_processed = processed_dfs[1:]
    train_df_processed = preprocess_dfs([train_df], preprocessing_settings)[0]

    # Prepare Model
    match ml_algorithm:
        case 'KNN':
            model = KNN_model_setup(base_parameter, param_settings)
        case 'SVM':
            model = SVM_model_setup(base_parameter, param_settings)
        case _:
            raise ValueError("Unsupported ML algorithm")
        
    if get_self_score:
        fitted_model, score, cv_scores = std_split_fit_and_scores\
        (train_df_processed, model, cv = cross_validation)
        general_fields = get_general_fields(nr_qubits, machines, tr_val_exp_type, tr_val_circuits, tr_val_exp_type, tr_val_circuits, preprocessing_settings)
        ml_fields = get_ml_fields(ml_algorithm, model.get_params(), param_settings)
        if cross_validation:
            results_fields = get_results_fields(score, cv_scores)
        else:
            results_fields = get_results_fields(score)
        ml_results_to_csv(general_fields, ml_fields, results_fields, filename, fields)


    # Fitted model
    X_train, Y_train = get_x_y(train_df_processed)
    Y_train = Y_train.to_numpy().ravel()
    fitted_model = model.fit(X_train, Y_train)
    # fitted_model = model.fit(X_train, Y_train.values.ravel())
    
    for test_df in test_dfs:
        test_df_processed = preprocess_dfs([test_df], preprocessing_settings)[0]
        X_test, Y_test = get_x_y(test_df_processed)
        test_score = fitted_model.score(X_test, Y_test)
        
        test_circuits = get_circuit_binary_from_df(test_df)
        #print(test_circuits)
        test_exp_type = test_df['experiment_type'].iloc[0]

        general_fields = get_general_fields(nr_qubits, machines, tr_val_exp_type, tr_val_circuits, test_exp_type, test_circuits, preprocessing_settings)
        ml_fields = get_ml_fields(ml_algorithm, model.get_params(), param_settings)
        results_fields = get_results_fields(test_score)
        ml_results_to_csv(general_fields, ml_fields, results_fields, filename, fields)


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

def get_HSR_array_all_backends(nr_qubits):

    df_H = dpf.get_expanded_df('Hardware',nr_qubits)
    df_S = dpf.get_expanded_df('Simulation',nr_qubits)
    df_R = dpf.get_expanded_df('Refreshed_Simulation',nr_qubits)

    return [df_H, df_S, df_R]

def get_circuit_type_array(df_nq):
    df_nqi = features_to_int(df_nq)
    circuits = split_into_circuits(df_nqi) 
    return circuits

def get_HSR_test_table(initial_list):
    
    list_of_arrays = generate_combos(initial_list)
    
    df_SR = pd.concat(initial_list[1:3].copy())
    
    # Quick and dirty fix for results_to_csv function:
    df_SR['experiment_type'] = 'Sim and Refreshed'

    #Train on H, Test on SR combined:
    list_of_arrays[0].append(df_SR)

    #make the train on H row only torino and brisbane:
    #H_backends = initial_list[0]['backend'].unique()
    for i in range(len(list_of_arrays)):
        backends = initial_list[i]['backend'].unique()
        list_of_arrays[i] = make_same_backends(list_of_arrays[i],backends)
    
    #Train on SR and Test on H only:
    train_SR_test_H = [df_SR,initial_list[0]]
    list_of_arrays.append(train_SR_test_H)

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

def get_accuracies_for_comparison(model, tr_val_dfp, tr_label,test_dfps, test_dfp_labels, to_print = False, get_self_score = True):
    
    test_scores = []
    labels =[]
    labels = labels +test_dfp_labels
    X_tr_val,Y_tr_val = get_x_y(tr_val_dfp)

    if get_self_score:
        labels.insert(0,"self_score")
        

        X_train_self, X_test_self, Y_train_self, Y_test_self = model_selection.train_test_split(
        X_tr_val,Y_tr_val,test_size=0.2,shuffle = True,random_state=42)

        fitted_model, self_score = fit_and_get_score(
        model,X_train_self,Y_train_self,X_test_self,Y_test_self)

        test_scores.append(self_score)
    
    Y_tr_val_1d = Y_tr_val.to_numpy()
    Y_tr_val_1d = Y_tr_val_1d.ravel()

    fitted_model_full = model.fit(X_tr_val,Y_tr_val_1d)

    for dfp in test_dfps:
        
        X,Y = get_x_y(dfp)
        test_score = fitted_model_full.score(X, Y) #check score vs accurcy_score
        test_scores.append(test_score)

    if to_print:
        print("Trained on ",tr_label)
        for i in range(len(test_scores)):
            print("test on ",labels[i],":",test_scores[i])

    return test_scores ,labels