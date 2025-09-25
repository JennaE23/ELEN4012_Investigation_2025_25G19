import data_extract_funcs
import circuit_funcs2
# import circuit_funcs

def get_and_save_sim_results(nr_qubits,fake_backend,nr_runs,output_file_names,create_csvs_,dir_):
    qc_set = circuit_funcs2.make_set_of_3(nr_qubits)
    # qc_set = circuit_funcs.make_set_of_3(nr_qubits)
    results_list = circuit_funcs2.run_sim(fake_backend,qc_set,nr_runs)
    # results_list = circuit_funcs.run_sim(fake_backend,qc_set,nr_runs)
    data_extract_funcs.sim_results_to_csv(nr_qubits,output_file_names,results_list,create_csvs_,dir_)

def run_sims_for_all_backends(fake_backends, nr_qubits,nr_runs,dir_,create_csvs_):
    for fake_backend in fake_backends:
        file_names = data_extract_funcs.make_file_names(fake_backend,str(nr_qubits))
        get_and_save_sim_results(nr_qubits,fake_backend,1,file_names,create_csvs_,dir_)
        for i in range(nr_runs-1):
            get_and_save_sim_results(nr_qubits,fake_backend,1,file_names,False,dir_)