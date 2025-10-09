from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeBrisbane
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh #fez, marrakesh are both Heron2
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

# Backend Lists
original_fake_backends = [FakeTorino(), FakeBrisbane()]
fake_backends = [FakeTorino(), FakeFez(), FakeMarrakesh(),FakeBrisbane()]
hardware_backends = [ service.backend('ibm_torino'),service.backend('ibm_brisbane')]

# Directory Lists
dir_Hardware_list = ["../Hardware_results/4q/","../Hardware_results/8q/"]
dir_Sims_list = ["../Simulated_results/4q/","../Simulated_results/8q/","../Simulated_results/16q/"]
dir_Refr_Sims_list = ["../Refreshed_Simulated_results/4q/","../Refreshed_Simulated_results/8q/","../Refreshed_Simulated_results/16q/"]

# Functions to update lists
def update_hardware_backends(hardware_backends_old, service_new):
    hardware_backends_new = hardware_backends_old
    new_backends = [service_new.backend('ibm_fez'),service_new.backend('ibm_marrakesh')]
    hardware_backends_new.extend(new_backends)
    return hardware_backends