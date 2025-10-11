from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from dataclasses import dataclass
import numpy as np

@dataclass
class SVM_mode:
    def __init__(
            self, 
            base_param_ = 'linear', 
            C_ =1, 
            coef0_=0, 
            degree_=3,
            label_ = 0
        ):
        self.model = SVC(
                kernel = base_param_, 
                C = C_, 
                degree = degree_, 
                coef0 = coef0_
            )
        self.base_param =base_param_# kernel
        self.C = C_
        self.coef0 = coef0_
        self.degree = degree_
        #self.model =model_
        self.label = label_
        self.__alg_type ='SVM'

    # #PUBLIC:
    # #initialisable
    # base_param : str = 'linear' # kernel
    # C: float =0
    # coef0: float = 0
    # degree: float =3
    # model : SVC = SVC(kernel = base_param, C = C, degree = degree, coef0 = coef0)
    # label : int =0
    def get_alg_type(self):
        return self.__alg_type
    # def get_model(self):
    #     return self.model
    
class KNN_mode:
    def __init__(self, base_param_=3, label_ =0):
        self.__alg_type = 'KNN'
        self.base_param = base_param_ #k
        self.model  = KNeighborsClassifier(
        n_neighbors=base_param_, 
        weights="distance",
        algorithm='auto', 
        leaf_size=30, 
        p=1
        )
        self.label =label_

    def get_KNN_models(self,k_s):
        models = []
        for i in k_s:
            models.append(KNN_mode(i))
        return models
    
    def get_alg_type(self):
        return self.__alg_type
    
class Param_Modes:  #modes or models?
    def __init__(self,knn_ks_ = range(3,26)):
    
        self.knn_ks = knn_ks_

        #generated
        self.SVM_modes = [
            #'mode0' : 
            SVM_mode(base_param_ = 'linear'),
            #'mode1' : 
            SVM_mode(base_param_ = 'poly',degree_ = 5, label_ =1),
            #'mode2' : 
            SVM_mode(base_param_ = 'rbf', C_ = 450,label_ =2),
            #'mode3' : 
            SVM_mode(base_param_ = 'poly', C_ = 450, coef0_ = 0.5,label_ =3),
            #'mode4' : 
            SVM_mode(base_param_ = 'poly', C_ = 450, degree_ = 5,label_ =4)
            #mode5 = SVC(kernel = 'linear', C = 450)
        ]

        self.KNN_modes = KNN_mode().get_KNN_models(knn_ks_)