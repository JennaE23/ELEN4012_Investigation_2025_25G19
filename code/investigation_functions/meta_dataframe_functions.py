import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# import qiskit
# from qiskit.providers.fake_provider import GenericBackendV2
# from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2, FakeTorino, FakeBrisbane
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh #fez, marrakesh are both Heron2
from qiskit_ibm_runtime import QiskitRuntimeService

from investigation_functions import data_extract_funcs

import backend_vars

#Call experiment_type as 'Hardware', 'Simulation', 'Refreshed_Simulation', 'Generic_Simulation'

def extract_cols_from_filename(file_name_str, dir_):
    file_path = dir_+file_name_str
    file_name_str = file_name_str.replace('_summarised','')
    nr_qubits = re.findall(r'\d+',file_name_str[0:3])[0]
    backend_name_start = len(nr_qubits)+2 #might need to 7 change if 'ibm_..." is included in file name"
    circuit_type = file_name_str[-5]
    is_sim = False
    
    if file_name_str.find("fake")!=-1:
        is_sim = True
    if is_sim:
        backend_name_start = backend_name_start +5
    else:
        backend_name_start = backend_name_start +4
        
    backend_name = file_name_str[backend_name_start:-5]
    
    return [nr_qubits,backend_name,is_sim,circuit_type,file_path]

def add_file_to_meta_df(meta_df,file_name,dir_):
    row = extract_cols_from_filename(file_name,dir_)
    meta_df.loc[len(meta_df)] = row
    
def add_qfolder_to_df(df,folder_dir,backend_names,nr_qubits, summarised =False):
    for backend_name in backend_names:
        #backend_name = backend.backend_name
        file_names = data_extract_funcs.make_file_names(backend_name,nr_qubits, summarised)
        for file_name in file_names:
            add_file_to_meta_df(df,file_name,folder_dir)

def add_qfolder_to_df_generic(df,folder_dir,backends_,nr_qubits):
    backend_name = backends_
    file_names = data_extract_funcs.make_file_names(backend_name,nr_qubits)
    for file_name in file_names:
        add_file_to_meta_df(df,file_name,folder_dir)

def add_qfolders(df,backend_names,folders, summarised = False):
    for folder in folders:
        nr_qubits = re.findall(r'\d+',folder)[0]
        add_qfolder_to_df(df,folder,backend_names,nr_qubits, summarised)

def add_qfolders_generic(df,backends,folders):
    for folder in folders:
        nr_qubits = re.findall(r'\d+',folder)[0]
        add_qfolder_to_df_generic(df,folder,backends,nr_qubits)

def blank_meta_df():
    meta_df = pd.DataFrame(columns = ['nr_qubits','backend','sim','circuit_type','file_path'])
    return meta_df

# Loads the particular data required. By default, loads brisbane and torino only for hardware and refreshed.
# To load full hardware*, set updated_results to 'True' and updated_service to your service.
# To load full refreshed*, set updated_results to 'True'.
# Note: A licensed service will be required to load full Hardware.
# *full refers to (torino, brisbane, fez and marrakesh), additional simulated results can be added be adjusting the backend_vars file.
def load_meta_df(
        meta_df,experiment_type, exp_dir_='',
        updated_results = False, updated_service = 'Default',
        summarised = False
    ):#Hardware,Simulation,Refreshed_Simulation
    dir_Hardware = backend_vars.make_dir_list(
        exp_dir_,
        backend_vars.Hardware_folder,
        incl_16=True
    )
    dir_Refr_Hardware = backend_vars.make_dir_list(
        exp_dir_,
        backend_vars.Refr_Hardware_folder
    )
    dir_Sims = backend_vars.make_dir_list(
        exp_dir_,
        backend_vars.Sim_folder,
        incl_16=True
    )
    dir_Refr_Sims = backend_vars.make_dir_list(
        exp_dir_,
        backend_vars.Refr_Sim_folder,
        incl_16=True
    )

    backend_names_ =[]
    dir_ =[]
    match experiment_type:
        case 'Refreshed_Hardware':
            dir_ = dir_Refr_Hardware
            backend_names_ = backend_vars.refr_hardware_backend_names
        case 'Hardware':
            dir_ = dir_Hardware
            hardware_backend_names = backend_vars.hardware_backend_names
            if updated_results:
                hardware_backend_names = backend_vars.all_hardware_backend_names
            backend_names_ = hardware_backend_names
        case 'Simulation':
            dir_ = dir_Sims
            backend_names_ = backend_vars.fake_backend_names
        case 'Refreshed_Simulation':
            dir_ = dir_Refr_Sims
            if updated_results:
                backend_names_ = backend_vars.fake_backend_names
            else:
                backend_names_ = backend_vars.original_fake_backend_names
        case 'Generic_Simulation':
            dir_ = dir_Sims
            backend_names_ = "fake_genericV2"
            add_qfolders_generic(meta_df, backend_names_,dir_)
            return

    add_qfolders(meta_df,backend_names_,dir_,summarised)

def get_results_df_from_row(row_index,meta_df):
    csv_file = meta_df.loc[row_index,'file_path']
    csv_df = pd.read_csv(csv_file)
    return csv_df

def add_df_column(meta_df):
    meta_df['df'] = meta_df['file_path'].apply(pd.read_csv)
    return meta_df
def get_percentage_non_zero(df):
    df.replace(0,'')
    percentage_non_zero = 100*sum(df.count())/(df.size)
    return percentage_non_zero

def add_sparsity_column(meta_df): #df column must already exist
    meta_df['percent_non_zero']=meta_df['df'].apply(get_percentage_non_zero)
    return meta_df

def get_experiment_type(file_path):
    if 'Hardware' in file_path:
        exp_type = 'Hardware'
        return exp_type
    if 'Refreshed' in file_path:
        exp_type = 'Refreshed Sim'
        return exp_type
    if 'Simulated' in file_path:
        exp_type = 'Sim'
        return exp_type

def add_experiment_type_column(meta_df):

    meta_df['experiment_type']=meta_df.loc[:,'file_path'].apply(get_experiment_type)
    return meta_df

