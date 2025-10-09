import csv
from csv import DictWriter
from qiskit_ibm_runtime import QiskitRuntimeService

def make_file_names(backend,nr_qubits):
    nr_qubits = str(nr_qubits)
    base_name = nr_qubits +"q_"
    base_name = base_name +backend

    file_names = []
    for i in range(1,4):
        temp_name = base_name + str(i) +".csv"
        file_names.append(temp_name)
    return file_names

def create_fields(nr_qubits):
    fields = []
    for i in range(2**nr_qubits):
        binary_str = format(i, '0' + str(nr_qubits) + 'b')
        fields.append(binary_str)
    return fields

# print(create_fields(2))
def create_csv(csv_file_name, fields):
    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fields)

def add_dir_to_filenames(dir_,file_names):
    file_names_new =[]
    for i in range(len(file_names)):
        file_names_new.append(dir_ + file_names[i])
    return file_names_new

def create_csvs(file_names_array, fields):
    for file_name in file_names_array:
        create_csv(file_name,fields)

def results_to_csv(csv_file_names, fields, job_id_file, service_):
    service = service_
    count = 0
    jobs_torino1 = []
    jobs_torino2 = []
    jobs_torino3 = []
    jobs_brisbane1 = []
    jobs_brisbane2 = []
    jobs_brisbane3 = []
    with open(job_id_file, 'r') as f:
        for job_id in f.readlines():
            if count != 2:
                #job_id = f.readline()
                job_id = job_id[:-1] #gets rid of extra blank space character
                #print(job_id)
                #print(len(job_id))
                job = service.job(job_id)
                for i in range(3):
                    #result =i
                    result = job.result()[i].data.meas.get_counts()
                    #print(result)
                    if i == 0:
                        circuit1 = result
                    elif i == 1:
                        circuit2 = result
                    else:
                        circuit3 = result
                if count == 0:
                    jobs_torino1.append(circuit1)
                    jobs_torino2.append(circuit2)
                    jobs_torino3.append(circuit3)
                elif count == 1:
                    jobs_brisbane1.append(circuit1)
                    jobs_brisbane2.append(circuit2)
                    jobs_brisbane3.append(circuit3)
            count = (count + 1) % 3
            #print(count)

        count2 = 0
        for csv_file_name in csv_file_names:
            match count2:
                case 0:
                    rows = jobs_torino1
                case 1:
                    rows = jobs_torino2    
                case 2:
                    rows = jobs_torino3
                case 3:
                    rows = jobs_brisbane1
                case 4:
                    rows = jobs_brisbane2
                case 5:
                    rows = jobs_brisbane3
            with open(csv_file_name, 'a', newline='') as f:
                writer = DictWriter(f, fieldnames=fields)
                writer.writerows(rows)
            count2 += 1

def results_to_csv2(nr_qubits,dir_,file_name_array,job_ids_file,service_,create_csvs_ = False):
    file_name_array2 =add_dir_to_filenames(dir_,file_name_array)
    fields_ = create_fields(nr_qubits)
    if create_csvs_:
        create_csvs(file_name_array2,fields_)
    results_to_csv(file_name_array2,fields_,job_ids_file,service_)

####################################################################
#For Simulated data:
def get_circuit_type_results(results_list, circuit_type):#circuit_type is 1,2 or 3
    circuit_results =[]
    #re-arrange results into each type of circuit
    for i in range(len(results_list)):
        circuit_results.append(results_list[i][circuit_type-1])
    return circuit_results

def sim_results_to_csv(nr_qubits,file_name_array,results_list,create_csvs_,dir_):
    fields_ = create_fields(nr_qubits)
    file_name_array2 =add_dir_to_filenames(dir_,file_name_array)
    fields_ = create_fields(nr_qubits)
    if create_csvs_:
        create_csvs(file_name_array2,fields_)
    for i in range(len(file_name_array)):
        rows = get_circuit_type_results(results_list,i+1)
        with open(file_name_array2[i], 'a', newline='') as f:
            writer = DictWriter(f, fieldnames=fields_)
            writer.writerows(rows)


