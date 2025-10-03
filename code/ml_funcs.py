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