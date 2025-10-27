import qiskit
from qiskit_ibm_runtime.fake_provider import  FakeTorino, FakeBrisbane
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh
from qiskit_ibm_runtime import QiskitRuntimeService

import sim_function

import config

print("Completed Imports")
fake_backends = [FakeFez(),FakeMarrakesh()]

# print("Refreshing Backends")
service = QiskitRuntimeService(channel = config.channel,token = config.tokenJAPI,instance = config.instanceJAPI)
# for backend in fake_backends:
#     backend.refresh(service)

# print("Refreshed Backends")

print("Specifying parameters")
dir_ = "Refreshed_Simulated_results/16q/"
create_csvs_ = False # if true, it creates and overwrites, if flase, it just appends
runs= 44
nr_qubits = 16

print("Running Sims for FakeFez")

sim_function.run_sims_for_all_backends([FakeFez()],nr_qubits,runs,dir_,create_csvs_)