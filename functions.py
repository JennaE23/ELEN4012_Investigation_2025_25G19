import csv
from csv import DictWriter

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

def results_to_csv(csv_file_names, fields, job_id_file):
    count = 0
    jobs_torino1 = []
    jobs_torino2 = []
    jobs_torino3 = []
    jobs_brisbane1 = []
    jobs_brisbane2 = []
    jobs_brisbane3 = []
    with open(job_id_file, 'r') as f:
        circuit1 = []
        circuit2 = []
        circuit3 = []
        if count != 2:
            job_id = f.readlines()
            job = service.job(job_id)
            for i in range(3):
                result = job.result()[i].data.meas.get_counts()
                if i == 0:
                    circuit1.append(result)
                elif i == 1:
                    circuit2.append(result)
                else:
                    circuit3.append(result)
            if count == 0:
                jobs_torino1.append(circuit1)
                jobs_torino2.append(circuit2)
                jobs_torino3.append(circuit3)
            elif count == 1:
                jobs_brisbane1.append(circuit1)
                jobs_brisbane2.append(circuit2)
                jobs_brisbane3.append(circuit3)
        count = (count + 1) % 3

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


    