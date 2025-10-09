from qiskit import QuantumCircuit 
from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

import datetime
import config

def make_set_of_3(nr_qubits):
    circuit_types = ['Cnot','Cnot_X','Swap']
    set_of_3 = []
    for i in range(len(circuit_types)):
        set_of_3.append(make_circuit(nr_qubits,circuit_types[i]))

    return set_of_3

def make_circuit(nr_qubits,circuit_type):
    # make circuit object
    qc = QuantumCircuit(nr_qubits)

    # apply hadamard gates to all qubits
    qc.h(range(nr_qubits))

    #left side of v
    qc = make_left_side(qc,nr_qubits,circuit_type)

    #right side of v
    qc = make_right_side(qc,nr_qubits,circuit_type)

    qc.h(range(nr_qubits))
    #apply measurement gates to all qubits
    qc.measure_all()

    return qc

def make_left_side(qc,nr_qubits,circuit_type):
    #right side of v
    for qubit in range(nr_qubits-1):
        match circuit_type:
            case 'Cnot':
                qc.cx(qubit,qubit+1)
            case 'Cnot_X':
                qc.cx(qubit,qubit+1)
                qc.x(qubit)
            case 'Swap':
                qc.swap(qubit,qubit+1)
    return qc

def make_right_side(qc,nr_qubits,circuit_type):
    #left side of v
    for qubit in range(nr_qubits-2,-1,-1): #start at nr_qubits-1, end at 0, step is -1
        match circuit_type:
            case 'Cnot':
                qc.cx(qubit,qubit+1)
            case 'Cnot_X':
                qc.x(qubit)
                qc.cx(qubit,qubit+1)
            case 'Swap':
                qc.swap(qubit,qubit+1)
    return qc

def run_job(backend_, circuit_set):
    #transpile circuits
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend_)
    isa_circuits = pm.run(circuit_set)
    sampler = Sampler(mode=backend_)
    sampler.options.environment.job_tags= config.tags
    #run job
    job = sampler.run(isa_circuits)

    #need code to save job ID
    return job.job_id()

def run_sim(fake_backend,qc_set, nr_runs):
    sim_backend = fake_backend
    # Transpile the ideal circuit to a circuit that can be directly executed by the backend
    transpiled_circuits = transpile(qc_set, sim_backend)
    results =[]
    # Run the transpiled circuit using the simulated backend
    for run in range(nr_runs):
        job = sim_backend.run(transpiled_circuits,shots =4096,memory=False)
        results.append(job.result().get_counts())

    return results

def send_set_to_backends(nr_qubits,backend_names,service_):
    service = service_
    #make set of circuits:
    qc_set = make_set_of_3(nr_qubits)
    job_IDs = []
    #send set to each backend
    for backend_name in backend_names:
        #store the job id for each backend job
        job_IDs.append(run_job(service.backend(backend_name),qc_set))
    return job_IDs

def send_and_record(nr_qubits,file_name,backend_names,service_):
    current_time = "datetime"+datetime.datetime.now().isoformat()
    lines = send_set_to_backends(nr_qubits,backend_names,service_)
    lines.append(current_time)
    text = "\n".join(lines) + "\n"
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(text)
    return lines