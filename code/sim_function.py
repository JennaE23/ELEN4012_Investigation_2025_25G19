import data_extract_funcs
import circuit_funcs

def get_and_save_sim_results(nr_qubits,fake_backend,nr_runs,output_file_names,create_csvs_,dir_):
    qc_set = circuit_funcs.make_set_of_3(nr_qubits)
    results_list = circuit_funcs.run_sim(fake_backend,qc_set,nr_runs)
    data_extract_funcs.sim_results_to_csv(nr_qubits,output_file_names,results_list,create_csvs_,dir_)