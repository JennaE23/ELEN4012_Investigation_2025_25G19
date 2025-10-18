import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from investigation_functions import  test_table_funcs as ttf
from investigation_functions import ml_funcs as mlf

fileName = "vizz_corr_HSR_c111.txt"
imageFileName = "vizz_corr_HSR_c111.png"
fileDir = ""
csvDir = "../"

file = open(fileDir+fileName, "w")

n_qubits_list = [4,8,16]
list_corr_HSR = []

for n_qubits in n_qubits_list:
    print("Getting correlations for "+str(n_qubits)+" qubits...")
    file.write("Qubit Size: "+str(n_qubits)+"\n")

    print("Getting initial list...")
    initial_list = ttf.get_HSR_array_all_backends(n_qubits,csvDir,True)
    print("Applying preprocessing...")
    df_H = mlf.apply_preprosessing(initial_list[0])
    df_S = mlf.apply_preprosessing(initial_list[1])
    df_R = mlf.apply_preprosessing(initial_list[2])

    print("Calculating correlations...")
    HS_corr_avg =df_H.corrwith(df_S,axis = 1).mean()
    HR_corr_avg =df_H.corrwith(df_R,axis = 1).mean()
    SR_corr_avg =df_R.corrwith(df_S,axis = 1).mean()

    list_corr_HSR.append([HS_corr_avg,HR_corr_avg,SR_corr_avg])

    file.write("Avg Correlation H vs S:\t"+str(HS_corr_avg)+"\n")
    file.write("Avg Correlation H vs R:\t"+str(HR_corr_avg)+"\n")
    file.write("Avg Correlation R vs S:\t"+str(SR_corr_avg)+"\n\n")
    print("Done for "+str(n_qubits)+" qubits.\n")

print("Plotting correlations...")
HS_corrs = [item[0] for item in list_corr_HSR]
HR_corrs = [item[1] for item in list_corr_HSR]
SR_corrs = [item[2] for item in list_corr_HSR]

fig = plt.figure()
plt.plot(qs,HS_corrs)
plt.plot(qs,HR_corrs)
plt.plot(qs,SR_corrs)
plt.legend(["HS","HR",'SR'])
plt.xlabel("Number of Qubits")
plt.ylabel("Average Correlation")
fig.savefig(imageFileName)
print("Done plotting.")
file.close()
