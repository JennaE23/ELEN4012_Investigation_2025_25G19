
from investigation_functions import  ml_to_csv_funcs as ml2csv
from investigation_functions import  ml_funcs as mlf
from investigation_functions import  test_table_funcs as ttf
import backend_vars
from ml_models_vars import Param_Modes

from itertools import combinations

qubits_list = backend_vars.qubits_list
svm_modes = Param_Modes().SVM_modes
knn_modes_all_k = Param_Modes().KNN_modes
knn_5_modes = []
for knn_modes in knn_modes_all_k:
    knn_5_modes.append(knn_modes[0])


dir_runs = ""
dir_ml = "ML_results/KNN/"
file_name = "knn_16q_HSR_c111.csv"
# file_name = "KNN_4q_test.csv"
ml2csv.create_ml_results_csv('KNN',dir_ml,file_name)

backend_list = ['brisbane','torino','fez','marrakesh']
backend_combos = list(combinations(backend_list, 2))
herons = tuple(['torino','fez','marrakesh'])
backend_combos.append(herons)

#to be changed if changing betw SVM and KNN:
param_modes = knn_5_modes
param_modes_best_H_S_R=param_modes[2:5]

qubit_nr = 16
initial_list = ttf.get_HSR_array_all_backends(qubit_nr,dir_runs, True)

ml2csv.run_and_record_HSR_c111(initial_list,dir_ml,file_name,param_modes)
# ml2csv.run_and_record_HSR_for_each_c_type(initial_list, dir_ml, file_name, param_modes)
# ml2csv.run_and_record_circuit_test_table(initial_list, dir_ml, file_name, param_modes)
# ml2csv.run_and_record_backends_v_backends(
#     initial_list, dir_ml, file_name,param_modes_best_H_S_R, backend_combos
# )