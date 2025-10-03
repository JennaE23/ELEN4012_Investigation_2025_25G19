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


def features_to_int(df):
    df_ = df
    label_encoder = LabelEncoder()
    df_['backend'] = label_encoder.fit_transform(df_['backend'])
    df_['circuit_type'] = pd.to_numeric(df_['circuit_type'], downcast='integer', errors='coerce')
    df_['nr_qubits'] = pd.to_numeric(df_['nr_qubits'], downcast='integer', errors='coerce')
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

def apply_preprosessing(df, includes_exp_type = True):#assumes only 1 nr of qubits
    df_ = df
    if includes_exp_type:
        df_ = df_.drop('experiment_type',axis = 1)
    df_ = features_to_int(df_)
    df_ = drop_0th_col(df_[['nr_qubits']].iloc[0],df_)
    df_ = df_.drop('nr_qubits', axis = 1)
    df_ = total_Err_to_percent(df_)
    return df_

def get_x_y(df_q,scale= True):
    Y = df_q[['backend']]
    X = df_q.drop('backend',axis = 1)
    if scale:
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