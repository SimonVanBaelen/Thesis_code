import os

import numpy as np
from dtaidistance import dtw
import sys
from src.data_loader import load_timeseries_from_tsv

def calculate(series,i,j,dm):
    if j > i:
        dm[i][j] = dtw.distance(series[i], series[j], use_c=True)
    elif j == i:
        dm[i][j] = 0
    else:
        dm[i][j] = dm[j][i]

def calculateAndSaveDM(start_i, name):
    path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
    _, series_train = load_timeseries_from_tsv(path_train)

    path_test = "Data/" + name + "/" + name + "_TEST.tsv"
    _, series_test = load_timeseries_from_tsv(path_test)
    series = np.concatenate((series_train, series_test), axis=0)

    filename = name + '_DM_nn.npy'
    try:
        print("starting from previous dm")
        dm = np.load(filename)
        if len(dm) < len(series):
            print("To short!", len(series)-len(dm))
            dm_to_short = np.zeros((len(series)-len(dm), len(series)))
            dm = np.concatenate((dm, dm_to_short), 0)
    except:
        print("starting from scratch")
        dm = np.zeros((len(series), len(series)))
        np.save(filename, dm)

    for i in range(start_i, len(series)):
        for j in range(0, len(series)):
            try:
                calculate(series, i, j, dm)
                print(i,j)
            except:
                if len(dm) < len(series):
                    dm_to_short = np.zeros((len(series) - len(dm), len(series)))
                    dm = np.concatenate((dm, dm_to_short), 0)
                    calculate(series, i, j, dm)
                    print(i, j)
                else:
                    raise Exception("something went wrong")
        if i % 1000 == 0:
            np.save(filename, dm)

    np.save(filename, dm)

# already_found_names = ["CBF","EthanolLevel", "ChlorineConcentration", "Wafer", "Crop","FiftyWords","FaceAll","ElectricDevices","ECG5000"]
all_names_to_do = ["WordSynonyms"]
start_i = 0
for name in all_names_to_do:
    calculateAndSaveDM(start_i, name)
    start_i = 0
