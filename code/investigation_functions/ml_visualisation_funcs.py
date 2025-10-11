import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import ml_funcs as mlf

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