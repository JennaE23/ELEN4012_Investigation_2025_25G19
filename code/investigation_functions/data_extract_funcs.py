import csv
from csv import DictWriter
#from qiskit_ibm_runtime import QiskitRuntimeService

def make_file_names(backend,nr_qubits):
    nr_qubits = str(nr_qubits)
    base_name = nr_qubits +"q_"
    base_name = base_name +backend

    file_names = []
    for i in range(1,4):
        temp_name = base_name + str(i) +".csv"
        file_names.append(temp_name)
    return file_names

def make_file_names_multi_backends(backend_names,nr_qubits):
    file_names_ = []
    for backend_name in backend_names:
        file_names_.append(make_file_names(backend_name,nr_qubits))
    return file_names_

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

def add_dir_to_filenames_list(dir_,file_names):
    file_names_new = file_names.copy()
    for backend_files in file_names_new:
        for i in range(len(backend_files)):
            backend_files[i]= dir_ + backend_files[i]
    return file_names_new

def add_dir_to_filenames_arr(dir_,file_names):
    file_names_new =[]
    for i in range(len(file_names)):
        file_names_new.append( dir_ + file_names[i])
    return file_names_new

def create_csvs(file_names_array, fields):
    for file_name in file_names_array:
        create_csv(file_name,fields)

def create_csvs_from_list(file_names_list, fields):
    for file_names_array in file_names_list:
        for file_name in file_names_array:
            create_csv(file_name,fields)

def results_to_csv( csv_file_names,fields, job_id_file, service_):

    with open(job_id_file, 'r') as f:
        backend_count = 0
        for job_id in f.readlines():
            
            # this skips the datetime line
            if "datetime" in job_id:
                #print("date_row")
                backend_count =0
                continue
            
            csv_backend_files = csv_file_names[backend_count]
            job_id = job_id[:-1] #gets rid of extra blank space character
          
            job = service_.job(job_id)
            for i in range(3):
                
                result = job.result()[i].data.meas.get_counts()

                with open(csv_backend_files[i], 'a', newline='') as f:
                    writer = DictWriter(f, fieldnames=fields)
                    writer.writerows([result])
            
            backend_count = backend_count+1
               

def results_to_csv2(backend_names, nr_qubits,dir_,job_ids_file,service_ ,create_csvs_ = False):
    file_name_list = make_file_names_multi_backends(backend_names,nr_qubits)
    file_name_list2 =add_dir_to_filenames_list(dir_,file_name_list)
    fields_ = create_fields(nr_qubits)
    if create_csvs_:
        create_csvs_from_list(file_name_list2,fields_)
    results_to_csv(file_name_list2,fields_,job_ids_file,service_)

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
    file_name_array2 =add_dir_to_filenames_list(dir_,file_name_array)
    #fields_ = create_fields(nr_qubits)
    if create_csvs_:
        create_csvs(file_name_array2,fields_)
    for i in range(len(file_name_array)):
        rows = get_circuit_type_results(results_list,i+1)
        with open(file_name_array2[i], 'a', newline='') as f:
            writer = DictWriter(f, fieldnames=fields_)
            writer.writerows(rows)


