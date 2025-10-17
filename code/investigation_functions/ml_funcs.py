import pandas as pd
import numpy as np

from sklearn import model_selection
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# from investigation_functions import data_process_funcs as dpf

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

def apply_preprosessing(
        df, drop_exp_type = True,label_encode = True,
        drop_q = True,
    ):#assumes only 1 nr of qubits
    df_ = df
    if drop_exp_type:
        df_ = df_.drop('experiment_type',axis = 1)
    if label_encode:
        df_ = label_encode_backend(df_)
    if drop_q:
        df_ = df_.drop('nr_qubits', axis = 1)
        
    df_ = features_to_int(df_)
    df_ = drop_0th_col(df_[['nr_qubits']].iloc[0],df_)
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