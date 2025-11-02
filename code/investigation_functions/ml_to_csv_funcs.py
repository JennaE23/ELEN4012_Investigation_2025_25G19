
from investigation_functions import ml_funcs as mlf
from investigation_functions import test_table_funcs as ttf

import pandas as pd
import numpy as np
#from sklearn.model_selection import cross_val_score
from csv import DictWriter
#/////////////////////////////////////////////////////////////
#csv things
def create_ml_results_csv(ml_alg, dir = '../ML_Results/', file_name = None):
    general_fields = ['nr_qubits','machines','tr&v exp_type','tr&v circuits', 'test exp_type','test circuits','preprocess settings']
    score_fields = ['accuracy','cv_1','cv_2','cv_3','cv_4','cv_5']
    if file_name is None:
        f_name = ml_alg + '_results.csv'
    else:
        f_name = file_name
    if ml_alg == 'SVM':
        file_name = dir + f_name
        ml_param_fields = ['kernal', 'param settings']
    else:
        file_name = dir + f_name
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

def get_machine_binary(machine_list, all_machines = ['torino', 'brisbane', 'fez', 'marrakesh']):
    binary_list = [1 if machine in machine_list else 0 for machine in all_machines]
    binary = "".join(str(x) for x in binary_list)
    return binary

def get_machine_binary_from_df(df, all_machines = ['torino', 'brisbane', 'fez', 'marrakesh']):
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
    # print(circuit_list)
    circuit_list = [str(i) for i in circuit_list]  # Ensure all elements are strings
    binary_list = [1 if circuit in circuit_list else 0 for circuit in all_circuits]
    binary = "".join(str(x) for x in binary_list)
    return binary

#/////////////////////////////////////////////////////////////
# Big Function + baby funcs
def preprocess_dfs(dfs, preprocessing_settings):
    # Example preprocessing based on settings
    # print("Preprocessing settings:", preprocessing_settings)
    processed_dfs = []
    for df in dfs:
        if preprocessing_settings==0:
            df_processed = mlf.apply_preprosessing(df) 
        else:
            raise ValueError("Unsupported preprocessing setting")

        processed_dfs.append(df_processed)
        # Add more preprocessing options as needed
    return processed_dfs


def get_file_name_and_fields(ml_algorithm, dir = '../ML_Results/', file_name = None):
    general_fields = ['nr_qubits','machines','tr&v exp_type','tr&v circuits', 'test exp_type','test circuits','preprocess settings']
    score_fields = ['accuracy','cv_1','cv_2','cv_3','cv_4','cv_5']
    if file_name is None:
        f_name = ml_algorithm + '_results.csv'
    else:
        f_name = file_name
    if ml_algorithm == 'SVM':
        file_name = dir + f_name
        ml_param_fields = ['kernal', 'param settings']
    elif ml_algorithm=='KNN':
        file_name = dir + f_name
        ml_param_fields = ['n_neighbors', 'param settings']
    else:
        raise ValueError("Unsupported ML algorithm")
    fields = general_fields + ml_param_fields + score_fields
    return file_name, fields

def run_and_print_ml_results(train_df,test_dfs,param_mode, dir = '../ML_Results/', get_self_score = True, preprocessing_settings = 0, cross_validation = False, file_name = None):
    
    ml_algorithm = param_mode.get_alg_type()

    # Get CSV setup
    nr_qubits = train_df['nr_qubits'].iloc[0]
    machines = get_machine_binary_from_df(train_df)
    tr_val_circuits = get_circuit_binary_from_df(train_df)
    tr_val_exp_type = train_df['experiment_type'].iloc[0]
    filename, fields = get_file_name_and_fields(ml_algorithm, dir, file_name)
    
    train_df_processed = preprocess_dfs([train_df], preprocessing_settings)[0]

    # Prepare Model
    model = param_mode.model
    param_settings = param_mode.label
    
    if get_self_score:
        fitted_model, score, cv_scores = mlf.std_split_fit_and_scores\
        (train_df_processed, model, cv = cross_validation)
        general_fields = get_general_fields(nr_qubits, machines, tr_val_exp_type, tr_val_circuits, tr_val_exp_type, tr_val_circuits, preprocessing_settings)
        ml_fields = get_ml_fields(ml_algorithm, model.get_params(), param_settings)
        if cross_validation:
            results_fields = get_results_fields(score, cv_scores)
        else:
            results_fields = get_results_fields(score)
        ml_results_to_csv(general_fields, ml_fields, results_fields, filename, fields)


    # Fitted model
    X_train, Y_train = mlf.get_x_y(train_df_processed)
    Y_train = Y_train.to_numpy().ravel()
    fitted_model = model.fit(X_train, Y_train)
    # fitted_model = model.fit(X_train, Y_train.values.ravel())
    
    for test_df in test_dfs:
        test_df_processed = preprocess_dfs([test_df], preprocessing_settings)[0]
        X_test, Y_test = mlf.get_x_y(test_df_processed)
        test_score = fitted_model.score(X_test, Y_test)
        
        test_circuits = get_circuit_binary_from_df(test_df)
        #print(test_circuits)
        test_exp_type = test_df['experiment_type'].iloc[0]

        general_fields = get_general_fields(nr_qubits, machines, tr_val_exp_type, tr_val_circuits, test_exp_type, test_circuits, preprocessing_settings)
        ml_fields = get_ml_fields(ml_algorithm, model.get_params(), param_settings)
        results_fields = get_results_fields(test_score)
        ml_results_to_csv(general_fields, ml_fields, results_fields, filename, fields)

def run_and_record_test_table_for_mode(test_table,param_mode, dir='../ML_Results/', file_name = None):
        # param_mode must be either an SVM_mode or a KNN_mode object
        # base_param = param_mode.base_param
        # alg_type = param_mode.alg_type

        for i in range(len(test_table)):

            train_df = test_table[i][0]
            # nr_test_dfs = len(test_table_HSR4q[i])
            test_dfs = test_table[i][1:]
            #print_test_table([test_table_HSR4q[i]],circ_types=False)
            
            run_and_print_ml_results(train_df,test_dfs,param_mode,dir = dir,cross_validation=True, file_name = file_name)


#initial_list = ttf.get_HSR_array_all_backends(nr_qubits,dir_runs, True)

def run_and_record_HSR_c111(initial_list_H_S_R,dir_ml,file_name,param_modes):

    test_table_HSR = ttf.get_HSR_test_table(initial_list_H_S_R)
    for mode in param_modes:
        run_and_record_test_table_for_mode(
            test_table_HSR,mode,dir_ml,file_name
        )

def run_and_record_HSR_for_each_c_type(initial_list_H_S_R, dir_ml, file_name, param_modes):
    #HSR for one circuit type at a time ('001','010','100')
    circuit_options = ['1','2','3']
    
    for circuit_type in circuit_options:

        init_list_circ =[]
        for exp_type in initial_list_H_S_R:
            init_list_circ.append(exp_type[exp_type['circuit_type']==circuit_type])

        test_table_HSR_c = ttf.get_HSR_test_table(init_list_circ)
        for mode in param_modes:
            run_and_record_test_table_for_mode(
                test_table_HSR_c,mode,dir_ml,file_name
            )
def run_and_record_HSR_train_pairs(initial_list_H_S_R, dir_ml, file_name, param_modes):
    # df_SR = pd.concat(initial_list[1:3].copy())
    # df_HS = pd.concat(initial_list[0:2].copy())
    # df_HR = pd.concat([initial_list[0],initial_list[2]])
    # df_HSR = pd.concat([df_HS,initial_list[2]])
    # # Quick and dirty fix for results_to_csv function:
    # df_SR['experiment_type'] = 'Sim and Refreshed'
    # df_HS['experiment_type'] = 'Hardware and Sim'
    # df_HR['experiment_type'] = 'Hardware and Refreshed'
    # df_HSR['experiment_type'] = 'Hardware, Sim, and Refreshed'
    # list_of_arrays.append([df_HS,initial_list[2]])
    # list_of_arrays.append([df_HR,initial_list[1]])
    # list_of_arrays.append([df_SR,initial_list[0]])
    # list_of_arrays.append([df_HSR,df_HSR])
    for mode in param_modes:
        for test_exp_type in range(len(initial_list_H_S_R)):
            df_combined_train = pd.concat(
                [initial_list_H_S_R[(1-test_exp_type)%3],
                initial_list_H_S_R[(2-test_exp_type)%3]]
            )
            # print(df_combined_train['experiment_type'].unique())
            exps = df_combined_train.loc[:,'experiment_type'].unique()
            # print(exps)
            df_combined_train.loc[:,'experiment_type'] = " ".join(exps)
            df_test =initial_list_H_S_R[test_exp_type]

            run_and_record_test_table_for_mode(
                [[df_combined_train,df_test]],
                mode,dir_ml,file_name
            )

def run_and_record_circuit_test_table(initial_list_H_S_R, dir_ml, file_name, param_modes):
    for exp_type in initial_list_H_S_R:
        test_table_circuits = ttf.get_circuits_test_table(exp_type)
        for mode in param_modes:
            run_and_record_test_table_for_mode(
                test_table_circuits,mode,dir_ml,file_name
            )

def run_and_record_backends_v_backends(
        initial_list_H_S_R, dir_ml, 
        file_name, param_modes_best_H_S_R, backend_combos_list,
        cross_val = True
    ):
    # svm_modes_for_exp_type = svm_modes[2:5]
    modes_for_exp_type = param_modes_best_H_S_R  
     
    for exp_type,mode in zip(initial_list_H_S_R,modes_for_exp_type):
        
        #split_into_backends
        for backend_combo in backend_combos_list:
            self_test_b = exp_type[exp_type['backend'].isin(backend_combo)]
            run_and_print_ml_results(
                self_test_b,[self_test_b],mode,
                dir_ml, cross_validation=cross_val,file_name=file_name
            )

def run_and_record_bvb_for_each_c_type(
        initial_list_H_S_R, dir_ml, file_name, param_modes_best_HSR,
        backend_combos_list,
        cross_val = True
    ):
    #HSR for one circuit type at a time ('001','010','100')
    circuit_options = ['1','2','3']
    
    for circuit_type in circuit_options:

        init_list_circ =[]
        for exp_type in initial_list_H_S_R:
            init_list_circ.append(exp_type[exp_type['circuit_type']==circuit_type])

        run_and_record_backends_v_backends(
            init_list_circ, dir_ml, 
            file_name, param_modes_best_HSR, backend_combos_list,
            cross_val
        )
        