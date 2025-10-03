import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection, datasets, svm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import data_process_funcs
import meta_dataframe_functions

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
    label_encoder = LabelEncoder()
    df_['backend'] = label_encoder.fit_transform(df_['backend'])
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
    return df_

def get_x_y(df_q,apply_scaler= True):
    Y = df_q[['backend']]
    X = df_q.drop('backend',axis = 1)
    if apply_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X,Y

def fit_and_get_score(model,X_train,Y_train,X_test,Y_test,ravel = True, to_print = False):
    model_ = model
    if ravel:
        Y_train_1d = Y_train.to_numpy()
        Y_train_1d = Y_train_1d.ravel()
    else:
        Y_train_1d = Y_train

    model_.fit(X_train, Y_train_1d)
    Cscore = model_.score(X_test, Y_test)
    if to_print:
        print("Accuracy:", Cscore)

    return model_,Cscore

def get_cv_score(model, X_train,Y_train,folds = 5,ravel=True):
    model_ = model
    if ravel:
        Y_train_1d = Y_train.to_numpy()
        Y_train_1d = Y_train_1d.ravel()
    else:
        Y_train_1d = Y_train
    score = cross_val_score(model_, X_train, Y_train_1d, cv=folds, scoring='accuracy')
    print("Cross-validation accuracy: ",score)
    return score

def std_split_fit_and_scores(dfp,model,scale = True, test_size_ = 0.2,fold_ = 5,cv = True):
    X,Y = get_x_y(dfp,scale)

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
    binary_list = [1 if circuit in circuit_list else 0 for circuit in all_circuits]
    binary = "".join(str(x) for x in binary_list)
    return binary

#/////////////////////////////////////////////////
#comparison test things
def split_into_circuits(df_all_circuits):
    circuits = df_all_circuits.groupby('circuit_type')
    circuit_1 = circuits.get_group(1)
    circuit_2 = circuits.get_group(2)
    circuit_3 = circuits.get_group(3)
    return [circuit_1,circuit_2,circuit_3]

def generate_combos(individual_dfps,include_combined=False):
    nr_indiv = len(individual_dfps)
    combos =[]
    
    for i in range(nr_indiv):
        combo = individual_dfps
        combo.insert(0, combo.pop(i))
        if include_combined:
            #make elements joined as pairs
            pair_dfs = make_pairs(combo[1:])
            #append the paired elements
            combo = combo+ pair_dfs
        combos.append(combo)

    return combos

def make_pairs(indiv_dfs):
    pairs = list(combinations(indiv_dfs, 2))
    pair_dfs = []
    for pair in pairs:
        df = pd.concat(pair)
        pair_dfs.append(df)

    return pair_dfs

def get_accuracies_for_comparison(model, tr_val_dfp, tr_label,test_dfps, test_dfp_labels, to_print = False, get_self_score = True):
    
    test_scores = []
    labels =[]
    labels = labels +test_dfp_labels
    labels.insert(0,"self_score")
    tr_val_dfp = apply_preprosessing(tr_val_dfp)
    

    X_tr_val,Y_tr_val = get_x_y(tr_val_dfp)
    X_train_self, X_test_self, Y_train_self, Y_test_self = model_selection.train_test_split(
    X_tr_val,Y_tr_val,test_size=0.2,shuffle = True,random_state=42)
    if get_self_score:
        fitted_model, self_score = fit_and_get_score(
        model,X_train_self,Y_train_self,X_test_self,Y_test_self)
    
    Y_tr_val_1d = Y_tr_val.to_numpy()
    Y_tr_val_1d = Y_tr_val_1d.ravel()

    fitted_model_full = model.fit(X_tr_val,Y_tr_val_1d)

    test_scores.append(self_score)

    for dfp in test_dfps:
        dfp = apply_preprosessing(dfp)
        X,Y = get_x_y(dfp)
        test_score = fitted_model_full.score(X, Y) #check score vs accurcy_score
        test_scores.append(test_score)

    if to_print:
        print("Trained on ",tr_label)
        for i in range(len(test_scores)):
            print("test on ",labels[i],":",test_scores[i])

    return test_scores ,labels