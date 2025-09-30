import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection, datasets, svm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import data_process_funcs
import meta_dataframe_functions

from sklearn.model_selection import cross_val_score


def features_to_int(df):
    df_ = df
    label_encoder = LabelEncoder()
    df_['backend'] = label_encoder.fit_transform(df_['backend'])
    df_['circuit_type'] = pd.to_numeric(df_['circuit_type'], downcast='integer', errors='coerce')
    return df_

def drop_0th_col(nr_qubits,df):
    df_ = df
    col_name = '0'*nr_qubits
    df_.drop(col_name,axis=1)
    return df_

def total_Err_to_percent(df): #in place of scaling
    df_ = df
    df_['totalError']= df_['totalError'].div(4096)
    return df_

def apply_preprosessing(df):#assumes only 1 nr of qubits
    df_ = df
    df_ = features_to_int(df_)
    df_ = drop_0th_col(df_[0,'nr_qubits'],df_)
    df_ = df_.drop('nr_qubits', axis = 1)
    df_ = total_Err_to_percent(df_)

def get_x_y(df_q):
    Y = df_q[['backend']]
    X = df_q.drop('backend',axis = 1)
    return X,Y

def fit_and_get_score_(model,X_train,Y_train,X_test,Y_test,ravel = True):
    model_ = model
    if ravel:
        Y_train_1d = Y_train.to_numpy()
        Y_train_1d = Y_train_1d.ravel()
    else:
        Y_train_1d = Y_train

    model_.fit(X_train, Y_train_1d)
    Cscore = model_.score(X_test, Y_test)
    print("Accuracy:", Cscore)

    return model_,Cscore

def get_cv_score(model, X_train,Y_train,folds = 5,):
    model_ = model
    score = cross_val_score(model_, X_train, Y_train, cv=folds, scoring='accuracy')
    print("Cross-validation accuracy: ",score)
    return score