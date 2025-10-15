import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from investigation_functions import ml_funcs as mlf
import seaborn as sns
import numpy as np

def print_and_plot_svm_models(df_processed, models, model_names, df_name='', graph_type = 'bar',to_print = 'False'):
    scores = []
    
    for model in models:
        fitted_model,score,cv_score = mlf.std_split_fit_and_scores(df_processed,model)
        if to_print:
            print(f"Model={model}, Accuracy={score}, CV_Accuracy={cv_score.mean()}")
            print(f"CV_Scores={cv_score}")
        scores.append(cv_score.mean())
    
    plt.figure(figsize=(10,6))
    match graph_type:
        case 'line':
            plt.plot(model_names, scores, color='skyblue', marker='o', linestyle='-')
        case 'bar':
            plt.bar(model_names, scores, color='skyblue')
        case _:
            raise ValueError("graph_type must be 'line' or 'bar'")
    plt.xlabel('SVM Models')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title(df_name +'SVM Model Comparison')
    plt.ylim(0, 1.2)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
    
    # return scores

def print_and_plot_knn_model_range_neighbours(df_processed, k_values, graph_type = 'line'):
    scores = []
    # model_names = [f'KNN (k={k})' for k in k_values]
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, weights="distance",algorithm='auto', leaf_size=30, p=1)
        fitted_model,score,cv_score = mlf.std_split_fit_and_scores(df_processed,model)
        print(f"KNN Model (k={k}), Accuracy={score}, CV_Accuracy={cv_score.mean()}")
        # print(f"CV_Scores={cv_score}")
        scores.append(cv_score.mean())
    
    plt.figure(figsize=(10,6))
    match graph_type:
        case 'line':
            plt.plot(k_values, scores, color='lightgreen', marker='o', linestyle='-')
        case 'bar':
            plt.bar(k_values, scores, color='lightgreen')
        case _:
            raise ValueError("graph_type must be 'line' or 'bar'")
    plt.xlabel('Nr of Neighbours')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN Model Comparison')
    plt.ylim(0, 1.2)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()
    
    # return scores
def get_df_with_same(col1,col2,df, drop_same_cols = True):
    df_ = df[df[col1]==df[col2]]
    if drop_same_cols:
        df_ = df_.drop(col1,axis=1)
        df_ = df_.drop(col2,axis=1)
    return df_

def add_avg_cv_col(df):
    df_ = df
    df_.loc[:,'cv_avg']=df_.loc[:,'cv_1':'cv_5'].mean(axis =1)
    return df_

def drop_cvs(df):
    cvs = ['cv_1','cv_2','cv_3','cv_4','cv_5']
    for cv in cvs:
        df=df.drop(cv,axis = 1)
    return df

def one_hot_to_int(df):
    one_hots = ['machines','tr&v circuits','test circuits']
    for col in one_hots:
        df[col] = df[col].astype(str)
        df[col]=df[col].apply(lambda x: int(x, 2))
    return df

def drop_preproc(df):
    df_ = df.drop('preprocess settings',axis =1)
    return df_

def make_easy2plot(df,drop_nr_q = True, add_cv_avg = True):
    df_ = df
    if add_cv_avg:
        df_ = add_avg_cv_col(df_)
    df_ = drop_cvs(df_)
    df_ = drop_preproc(df_)
    if drop_nr_q:
        df_ = df_.drop('nr_qubits',axis = 1)
    return df_

def plot_bar_per_qubit_nr(
        df4q,df8q,df16q, labels_,nr_cat =3,x_='machines',y_='accuracy',lowerY=0,
        hue_ = 'tr&v exp_type'):
    
    plt.subplot(311)
    ax_4qs =sns.barplot(
        df4q, x = x_, y = y_,
        hue = hue_)
    ax_4qs.set_ylim(tuple([lowerY,1]))
    ax_4qs.set_xticks(ticks = np.arange(0,nr_cat),labels=labels_)

    plt.subplot(312)
    ax_8qs=sns.barplot(
        df8q, x = x_, y = y_,
        hue = hue_)
    ax_8qs.set_ylim(tuple([lowerY,1]))
    ax_8qs.set_xticks(ticks = np.arange(0,nr_cat),labels=labels_)

    plt.subplot(313)
    ax_16qs=sns.barplot(
        df16q, x = x_, y = y_,
        hue = hue_)
    ax_16qs.set_ylim(tuple([lowerY,1]))
    ax_16qs.set_xticks(ticks = np.arange(0,nr_cat),labels=labels_)

    plt.show()