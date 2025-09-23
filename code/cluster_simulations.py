import qiskit
from qiskit import QuantumCircuit #Aer, IBMQ,
from qiskit.visualization import plot_histogram
from qiskit import transpile
#from qiskit.providers import fake_provider
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeTorino, FakeBrisbane
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh
#fez, marrakesh are both Heron2
from qiskit.providers.basic_provider import BasicSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# importing datetime module for now()
# import datetime
# import matplotlib

# import circuit_funcs
# import data_extract_funcs
import sim_function

def make_file_names(backend_name,is_sim,nr_qubits):
    base_name = nr_qubits +"q_"
    if is_sim:
        base_name = base_name +"fake_"
    base_name = base_name +backend_name

    file_names = []
    for i in range(1,4):
        temp_name = base_name + str(i) +".csv"
        file_names.append(temp_name)
    return file_names

def run_sims_for_all_backends(fake_backends, nr_qubits,nr_runs,dir_,create_csvs_):
    for fake_backend in fake_backends:
        file_names = make_file_names(fake_backend.backend_name,False,str(nr_qubits))
        sim_function.get_and_save_sim_results(nr_qubits,fake_backend,1,file_names,create_csvs_,dir_)
        for i in range(nr_runs-1):
            sim_function.get_and_save_sim_results(nr_qubits,fake_backend,1,file_names,False,dir_)

fake_backends = [FakeTorino(),FakeFez(),FakeMarrakesh(),FakeBrisbane()]

dir_ = "../Simulated_results/4qCluster/"
create_csvs_ = True # if true, it creates and overwrites, if false, it just appends
runs= 10
nr_qubits = 4

run_sims_for_all_backends([FakeBrisbane()],nr_qubits,runs,dir_,create_csvs_)