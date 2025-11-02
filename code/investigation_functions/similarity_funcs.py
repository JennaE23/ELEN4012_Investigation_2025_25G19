import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from csv import DictWriter

import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_distances

from investigation_functions import data_process_funcs as dpf

def get_test_list(
        qubits_list,change,
        backends_list = ['brisbane','torino','fez','marrakesh'],
        circuits_list = ['1','2','3'],
        exp_type_list = ['Hardware', 'Simulated','Refreshed_Simulated']
    ):
    
    backend_same =list(zip(backends_list,backends_list))
    circuit_same =list(zip(circuits_list,circuits_list))
    exp_type_same=list(zip(exp_type_list,exp_type_list))

    exp_type_pairs_ = exp_type_same
    circuit_pairs_ = circuit_same
    backend_pairs_ = backend_same

    if change == 'exp_type':
        exp_type_pairs_ =  list(combinations(exp_type_list, 2))
    elif change == 'circuits':
        circuit_pairs_ =  list(combinations(circuits_list, 2))
    elif change == 'backends':
        backend_pairs_ = list(combinations(backends_list, 2))
    else:
        raise('invalid change variable')
    

    test_combos = []

    
    for exp_type_pair in exp_type_pairs_:
    
        for nr_qubits in qubits_list:

            for circuit_pair in circuit_pairs_:

                for backend_pair in backend_pairs_:

                    row = {
                        'exp_type_pair':exp_type_pair,
                        'nr_qubits':nr_qubits,
                        'circuit_pair':circuit_pair,
                        'backend_pair':backend_pair
                    }
                    test_combos.append(row)
    
    return test_combos
def get_df_file_path_from_row(test_row, dir_runs, df_nr = 1,summarised = False):
    index = df_nr-1
    nq = str(test_row['nr_qubits'])
    exp_t =test_row['exp_type_pair'][index]
    circ = str(test_row['circuit_pair'][index])

    #get csv file_name
    file_name = nq + "q_"
    if exp_t =='Hardware':
        file_name = file_name +"ibm_"+test_row['backend_pair'][index]
    else:
        file_name = file_name +"fake_"+test_row['backend_pair'][index]
    if summarised:
        file_name = file_name +circ+"_summarised.csv"
    else:
        file_name = file_name +circ+".csv"

    #get file path
    file_path = dir_runs + exp_t +"_results/" 
    file_path = file_path+ nq+"q/"
    file_path = file_path + file_name

    return file_path


def load_test_dfs_from_test_row(test_row,dir_runs):
    df1_file_name = get_df_file_path_from_row(test_row,dir_runs,1)
    df2_file_name = get_df_file_path_from_row(test_row,dir_runs,2)

    df1 = pd.read_csv(df1_file_name)
    df2 = pd.read_csv(df2_file_name)

    return df1,df2

def load_summarised_test_dfs_from_test_row(test_row,dir_runs):
    df1_file_name = get_df_file_path_from_row(test_row,dir_runs,1,summarised=True)
    df2_file_name = get_df_file_path_from_row(test_row,dir_runs,2,summarised=True)

    df1 = pd.read_csv(df1_file_name)
    df2 = pd.read_csv(df2_file_name)
    return df1,df2

def get_and_record_corr_from_test_row(test_row, file_name,dir_corr,dir_runs):
    df1,df2 = load_test_dfs_from_test_row(test_row,dir_runs)
    corr_avg = df1.corrwith(df2).mean()
    
    fields = [
        "nr_qubits","exp_type 1","exp_type 2",
        "backend 1","backend 2","circuit 1","circuit 2",
        "corr avg"
    ]
    
    row = {
        "nr_qubits":test_row['nr_qubits'],
        "exp_type 1":test_row['exp_type_pair'][0],
        "exp_type 2":test_row['exp_type_pair'][1],
        "backend 1":test_row['backend_pair'][0],
        "backend 2":test_row['backend_pair'][1],
        "circuit 1":test_row['circuit_pair'][0],
        "circuit 2":test_row['circuit_pair'][1],
        "corr avg":corr_avg
    }
    with open(dir_corr+file_name, 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writerow(row)


def create_corr_csv(file_name,dir_corr):
    fields = [
        "nr_qubits","exp_type 1","exp_type 2",
        "backend 1","backend 2","circuit 1","circuit 2",
        "corr avg","cosine d","manhatt d","mse","euclidean d",
        "norm manhatt d","norm mse","norm euclidean d"
    ]
    with open(dir_corr+file_name, 'w', newline='') as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writeheader()

def get_and_record_corrs_from_tests_list(tests_list,file_name,dir_corr,dir_runs,create_csv = False):
    # fields = [
    #     "nr_qubits","exp_type 1","exp_type 2",
    #     "backend 1","backend 2","circuit 1","circuit 2",
    #     "corr avg"
    # ]
    if create_csv:
        create_corr_csv(file_name, dir_corr)
    
    for test_row in tests_list:
        get_and_record_corr_from_test_row(test_row,file_name,dir_corr,dir_runs)

def get_and_record_metrics_from_test_row(
        test_row, file_name,dir_corr,dir_runs,
        summarised = True,
        corrs =True,
        cosine_d = True,
        manhatt_d = True,
        mse_ = True,
        euclidean_d = True
    ):
    df1,df2 = load_test_dfs_from_test_row(test_row,dir_runs)
    if not summarised:
        df1,df2 = load_test_dfs_from_test_row(test_row,dir_runs)
        cosine_d = False
        manhatt_d = False
        mse_ = False
        euclidean_d = False
        corr_avg = df1.corrwith(df2,axis =1).mean() if corrs else None
    else:
        df1,df2 = load_summarised_test_dfs_from_test_row(test_row,dir_runs)
        df1 = df1['mean']
        df2 = df2['mean']
        corr_avg = df1.corr(df2) if corrs else None

    cos_d = cosine_distances([df1],[df2])[0][0] if cosine_d else None
    n_manh_d = norm_manhatt_dist(df1,df2) if manhatt_d else None
    n_mse = normalized_mse(df1,df2) if mse_ else None
    n_euc_d = norm_eucclidean_dist(df1,df2) if euclidean_d else None
    manh_d = manhatt_dist(df1,df2) if manhatt_d else None
    mse = mean_squared_error(df1,df2) if mse_ else None
    euc_d = np.linalg.norm(df1 - df2) if euclidean_d else None

    fields = [
        "nr_qubits","exp_type 1","exp_type 2",
        "backend 1","backend 2","circuit 1","circuit 2",
        "corr avg","cosine d","manhatt d","mse","euclidean d",
        "norm manhatt d","norm mse","norm euclidean d"
    ]
    
    row = {
        "nr_qubits":test_row['nr_qubits'],
        "exp_type 1":test_row['exp_type_pair'][0],
        "exp_type 2":test_row['exp_type_pair'][1],
        "backend 1":test_row['backend_pair'][0],
        "backend 2":test_row['backend_pair'][1],
        "circuit 1":test_row['circuit_pair'][0],
        "circuit 2":test_row['circuit_pair'][1],
        "corr avg":corr_avg,
        "cosine d":cos_d,
        "manhatt d":manh_d,
        "mse":mse,
        "euclidean d":euc_d,
        "norm manhatt d":n_manh_d,
        "norm mse":n_mse,
        "norm euclidean d":n_euc_d
    
    }
    with open(dir_corr+file_name, 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=fields)
        writer.writerow(row)

def get_and_record_metrics_from_tests_list(
        tests_list,file_name,dir_corr,dir_runs,create_csv = False,
        corrs =True,
        cosine_d = True,
        manhatt_d = True,
        mse = True,
        euclidean_d = True
        ):
    if create_csv:
        create_corr_csv(file_name, dir_corr)
    
    for test_row in tests_list:
        get_and_record_metrics_from_test_row(
            test_row,file_name,dir_corr,dir_runs,
            corrs,
            cosine_d,
            manhatt_d,
            mse,
            euclidean_d
        )
            

def rms(arr):
    rms_ = np.sqrt(np.mean(np.square(arr)))
    return rms_

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def normalized_mse(x1,x2):
    #https://www.mathworks.com/help/comm/ug/normalized-mean-square-distance-measure.html
    mse = mean_squared_error(x1, x2)
    denom = (x1.apply(lambda x: x**2)).mean()
    n_mse = mse/denom
    return n_mse

def norm_eucclidean_dist(x1,x2):
    x1n = normalize_vector(x1)
    x2n = normalize_vector(x2)
    return np.linalg.norm(x1n - x2n)

def manhatt_dist(x1,x2):
    return np.sum(np.abs(x1 - x2))

def norm_manhatt_dist(x1,x2):
    x1n = normalize_vector(x1)
    x2n = normalize_vector(x2)
    return np.sum(np.abs(x1n - x2n))

def combine_cols(corr_df):
    new_df = corr_df

    exps = new_df.loc[:,'exp_type 1':'exp_type 2'].apply(np.unique,axis=1)
    new_df['exp types'] = exps.apply(" ".join)

    circs = new_df.loc[:,'circuit 1':'circuit 2'].astype(str)
    circs = circs.apply(np.unique,axis=1)
    new_df['circuits'] = circs.apply(" ".join)

    backs = new_df.loc[:,'backend 1':'backend 2'].apply(np.unique,axis=1)
    new_df['backends'] = backs.apply(" ".join)

    new_df.drop(
        [
            'exp_type 1','exp_type 2','circuit 1','circuit 2',
            'backend 1','backend 2'
        ],
        axis=1,inplace = True)

    return new_df
    # ", ".join(my_tuple)
def add_corr_mag(df):
    df_ = df
    df_['corr avg mag']=df_['corr avg'].apply(abs)
    return df

def make_line_plots( 
        df,
        x= 'nr_qubits',
        y = 'corr avg',
        hue = 'circuits',
        col='backends',
        title_ = None,
        fig_size_ = (12,2),
        x_label = 'Number of Qubits',
        y_label = None,
        y_lim = None,
        share_y = True,
        share_y_ticks = True,
        grid = True,
        axis_font_size = 15,
        col_titles = None,
        legend_title = None,
        legend_labels = None,
        legend_fontsize = 14,
        vertical_stack = False,
        x_ticks_auto = False
    ):
    if y_label ==None:
        y_label = y
        
    cols = df[col].unique()
    num_cols = len(cols)
    num_rows =1
    if vertical_stack:
        num_cols = 1
        num_rows =len(cols)
    
    axs =[]
    if col_titles == None:
        col_titles = cols
    if legend_title == None:
        legend_title = hue
    
    fig = plt.figure(layout = 'constrained',figsize=fig_size_)
    # fig = plt.figure(figsize=fig_size_)
    fig.suptitle(title_, fontsize=16, fontweight='bold')
    for i in range(max(num_cols,num_rows)):
        if share_y and i>0:
            y_ax = axs[0]
            
        else:
            y_ax = None
        plt.subplot(num_rows,num_cols,i+1, sharey = y_ax,label = str(i))
        axs.append(
            sns.lineplot(
                data = df[df[col]==cols[i]],
                x= x,
                y = y,
                hue = hue
            )
        )
        axs[i].get_legend().remove()
        axs[i].set_title(col_titles[i], fontsize = axis_font_size+1)
        axs[i].set_ylabel(y_label, fontsize = axis_font_size)
        axs[i].set_xlabel(x_label, fontsize = axis_font_size)
        if not x_ticks_auto:
            axs[i].set_xticks([4,8,16])
        axs[i].tick_params(
            axis='both', which='both', labelsize=axis_font_size)
        axs[i].set_ylim(y_lim)
        if grid:
            axs[i].grid(visible =grid, linestyle ='dotted')   
    
    if share_y:
        for ax in axs[1:]:
            ax.set_ylabel('')
            # ax.set_yticks
   
    if share_y_ticks:
            for ax in axs:
                ax.label_outer()
 
    axs[1].legend(
        # handles =axs[0],
        title = legend_title,
        labels = legend_labels,
        title_fontsize = legend_fontsize+1,
        fontsize = legend_fontsize,
        # loc='center left', 
        # bbox_to_anchor=(1.05, 0.5), borderaxespad=0.
        loc='lower center', bbox_to_anchor=(0.5, -0.1),
        borderaxespad=10,
        ncol =3
        )
    return axs

def get_tot_err(file_path):
    # print(row)
    df = pd.read_csv(file_path)
    # nr_qubits = n_q
    # col_name_0th = str(np.strings.multiply('0',nr_qubits))
    tot_err = df.loc[0,'mean']
    return tot_err

def add_tot_err_col(df):
    df_ = df
    for i in range(len(df_)):
        # nq = int(df_.loc[i,'nr_qubits'])
        file_path = df_.loc[i,'file_path']
        df_.loc[i,'tot_err']=get_tot_err(file_path)
    return df_

def summarise_and_save_df(
        df, dir='../', exp_type_folder_name='Default', unique_dir = False, save_dir = ''):
    circuit_types = df['circuit_type'].unique()
    backends = df['backend'].unique()

    for backend in backends:
        for circuit in circuit_types:
            df_subset = df[(df['backend'] == backend) & (df['circuit_type'] == circuit)]
            df_ranged = dpf.find_range_dataframe(df_subset)

            if exp_type_folder_name != 'Default':
                exp_type = exp_type_folder_name
                exp_type=exp_type.replace("ion", "ed")
            else:
                exp_type = df_subset['experiment_type'].iloc[0]
                exp_type=exp_type.replace("ion", "ed")
            
            nr_qubits = df_subset['nr_qubits'].iloc[0]

            if exp_type != 'Hardware':
                backend_ = "fake_" + backend
            else:
                backend_ = "ibm_" + backend

            if unique_dir:
                file_name = f'{save_dir}{nr_qubits}q_{backend_}{circuit}_{exp_type}_summarised.csv'
            else:
                
                file_name  = f'{dir}{exp_type}_results/{nr_qubits}q/{nr_qubits}q_{backend_}{circuit}_summarised.csv'
            df_ranged.to_csv(file_name, index_label='column')

def summarise_and_save(experiment_type, nr_qubits, dir='../', exp_type_folder_name='Default', unique_dir = False, save_dir = ''):
    df = dpf.get_expanded_df(experiment_type, nr_qubits, dir_ = dir, processed = False, updated_results=True)
    summarise_and_save_df(df, dir, exp_type_folder_name, unique_dir, save_dir)

def plot_fingerprints(
    dfs,
    title_,
    rows_ylim = [10,5],
    backends = ['brisbane','torino','fez','marrakesh'],
    x_labels = 'Erroneous Outcomes',
    y_labels = 'Measured Probability \n (%)',
    n_rows = 2,
    n_cols = 2,
    x_label_size = 14,
    y_label_size = 14,
    subtitle_size = 16,
    subtitle_y = 1.05,
    x_tick_rot = 90,
    x_tick_size =14,
    sharey_by_row = True,
    show_plot = True,
    colour_wrap = 4,
    fig_size_ =(10,6)
    # share_x_all = True
    ):
    colours = [
        mcolors.CSS4_COLORS['darkturquoise'],
        mcolors.CSS4_COLORS['dodgerblue'],
        mcolors.CSS4_COLORS['blue'],
        mcolors.CSS4_COLORS['royalblue']
    ]
    fig = plt.figure(figsize=fig_size_, layout = 'constrained')

    plt.suptitle(
        title_,fontsize = 17,
        y = 1.08
    )
    axs =[]
    # ax_x = None

    for i in range(len(dfs)):
        
        axs_to_share = None
        if sharey_by_row:
            if (i+1)%n_cols==0:
                axs_to_share = axs[i-1]
        # if share_x_all and i>0:
        #     ax_x = axs[0]

        axs.append(plt.subplot(
            n_rows,n_cols,i+1,ymargin =1,sharey=axs_to_share))
        axs[i] =sns.barplot(
            data =dfs[i], label = backends[i],
            color=colours[i%colour_wrap]
        )
        axs[i].grid(linestyle='dotted')
        
        axs[i].set_ylabel(y_labels ,fontsize = y_label_size)
        axs[i].set_xlabel(x_labels ,fontsize = x_label_size)
        # ax1.get_xticks()
        axs[i].set_title(
            backends[i],fontsize = subtitle_size,y=subtitle_y)
        if i>n_cols and sharey_by_row:#only works for 2 rows
            axs[i].set_ylim((0,rows_ylim[1]))
        elif sharey_by_row:
            axs[i].set_ylim((0,rows_ylim[0]))
        plt.xticks(rotation = x_tick_rot,fontsize = x_tick_size)
        plt.yticks(fontsize = y_label_size)
    if show_plot:
        plt.show()
    return fig,axs

def __drop_0th(df,nr_qubits):
    col_name = np.strings.multiply('0',nr_qubits)
    df_no0 =df.drop(str(col_name),axis=1)
    return df_no0

def get_err_mean_percents(
        df_to_sort,
        nr_qubits=4,
        backends=['brisbane','torino','fez','marrakesh'],
        drop_0th = True
    ):
    # print(df_to_sort.info())
    if drop_0th:
        df_to_sort = __drop_0th(df_to_sort,nr_qubits)
    # print(df_to_sort.info())
    # dfs_split =[]
    means_percent =[]
    for backend in backends:
        df_b =df_to_sort[df_to_sort['backend']==backend]
        means_percent.append(dpf.find_mean_cols(df_b)*100/4096)
        # dfs_split.append(df_b)

    return means_percent

def get_means_above(means,above_val):
    means_above =[]
    for mean in means:
        high_mean =mean[mean>=0.5]
        means_above.append(high_mean)
        # print(high_mean)
    return means_above

def split_df_by_circuit_type(df):
    c1_df = df[df['circuit_type']=='1']
    c2_df = df[df['circuit_type']=='2']
    c3_df = df[df['circuit_type']=='3']
    return [c1_df,c2_df,c3_df]

def split_df_by_baceknds(df,baceknds=['brisbane','torino','fez','marrakesh']):
    dfs =[]
    for backend in baceknds:
        df_b = df[df['backend']==backend]
        dfs.append(df_b)
    return dfs