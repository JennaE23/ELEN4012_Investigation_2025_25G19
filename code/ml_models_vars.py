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
        self.__ALG_TYPE ='SVM'

    def get_alg_type(self):
        return self.__ALG_TYPE
    
    
class KNN_mode:
    def __init__(
            self, 
            base_param_=3,
            weights_="distance",
            algorithm_='auto', 
            leaf_size_=30, 
            p_=1,
            label_ =0
        ):
        self.__ALG_TYPE = 'KNN'
        self.base_param = base_param_ #k
        self.model  = KNeighborsClassifier(
        n_neighbors=base_param_, 
        weights=weights_,
        algorithm=algorithm_, 
        leaf_size=leaf_size_, 
        p=p_
        )
        self.label =label_

    def get_KNN_array(
            self,k_s,
            weights_="distance",
            algorithm_='auto', 
            leaf_size_=30, 
            p_=1,
            label_ =0
        ):
        models = []
        for i in k_s:
            models.append(KNN_mode(
                i,
                weights_,
                algorithm_, 
                leaf_size_, 
                p_,
                label_
            ))
        return models
    
    def get_alg_type(self):
        return self.__ALG_TYPE

    
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

        self.KNN_modes = [
            KNN_mode(base_param = 3, weights_='distance', label_=0),
            KNN_mode(base_param = 2, weights_='uniform', label_=1)
            #etc
        ]

    def add_SVM_mode(self,svm_mode):
        self.SVM_modes.append(svm_mode)
    
    def add_KNN_mode(self,knn_mode):
        self.KNN_modes.append(knn_mode)