import numpy as np
from dtaidistance import dtw

from src.data_loader import load_timeseries_from_tsv


def calculateAndSaveDM(start_i, name):
    path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
    _, series_train = load_timeseries_from_tsv(path_train)

    path_test = "Data/" + name + "/" + name + "_TEST.tsv"
    _, series_test = load_timeseries_from_tsv(path_test)
    series = series_train + series_test

    filename = name + '_DM_nn.csv'
    try:
        print("starting from previous dm")
        dm = np.loadtxt(filename, delimiter=',')
    except:
        print("starting from scratch")
        dm = np.zeros((len(series), len(series)))
        np.savetxt(filename, dm, delimiter=',')

    for i in range(start_i, len(series)):
        for j in range(0, len(series)):
            if j > i:
                dm[i][j] = dtw.distance(series[i], series[j], use_c=True)
            elif j == i:
                dm[i][j] = 0
            else:
                dm[i][j] = dm[j][i]
            print(i,j)
        np.savetxt(filename, dm, delimiter=',')