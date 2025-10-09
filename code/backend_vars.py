from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeBrisbane
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh #fez, marrakesh are both Heron2
from qiskit_ibm_runtime import QiskitRuntimeService

from investigation_functions import data_extract_funcs
service = QiskitRuntimeService()

# Backend Lists
original_fake_backends = [FakeTorino(), FakeBrisbane()]
fake_backends = [FakeTorino(), FakeFez(), FakeMarrakesh(),FakeBrisbane()]
hardware_backends = [ service.backend('ibm_torino'),service.backend('ibm_brisbane')]

# Qubits list
qubits_list = [4,8,16]

# Directory Lists
Hardware_folder = "Hardware_results/"
Sim_folder = "Simulated_results/"
Refr_Sim_folder = "Refreshed_Simulated_results/"

# Hardware_subfolders = get_exp_type_subfolders(Hardware_folder)
# Sim_subfolders = get_exp_type_subfolders(Sim_folder)
# Refr_Sim_subfolders = get_exp_type_subfolders(Refr_Sim_folder)
 

dir_Hardware_list = ["../Hardware_results/4q/","../Hardware_results/8q/"]
dir_Sims_list = ["../Simulated_results/4q/","../Simulated_results/8q/","../Simulated_results/16q/"]
dir_Refr_Sims_list = ["../Refreshed_Simulated_results/4q/","../Refreshed_Simulated_results/8q/","../Refreshed_Simulated_results/16q/"]

# Functions to update lists
def update_hardware_backends(hardware_backends_old, service_new):
    hardware_backends_new = hardware_backends_old
    new_backends = [service_new.backend('ibm_fez'),service_new.backend('ibm_marrakesh')]
    hardware_backends_new.extend(new_backends)
    return hardware_backends

def get_exp_type_subfolders(exp_folder, incl_16q = False):
    subfolders_dir =[]
    for qubit in qubits_list:
        if qubit == 16 and not incl_16q:
            continue
        subfolders_dir.append(exp_folder+str(qubit)+"q/")
    return subfolders_dir

def make_dir_list(exp_dir_,exp_folder,incl_16 = False):
    dir_list = data_extract_funcs.add_dir_to_filenames_arr(
        exp_dir_,
        get_exp_type_subfolders(exp_folder,incl_16)
    )
    return dir_list