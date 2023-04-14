from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
from dtaidistance import dtw, clustering
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

# FacesUCR, FaceAll
names = ["CBF"]

name = names[0]
path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
labels_train, series_train = load_timeseries_from_tsv(path_train)

path_test = "Data/" + name + "/" + name + "_TEST.tsv"
labels_tr, series_tr = load_timeseries_from_tsv(path_test)

func_name = "dtw"         # "msm"/"ed" other options
args = {"window": len(series_tr[0])-1}   # for MSM "c" parameter

def add_series_to_dm(true, next, dm):
    next = next - 1
    all_dtw = np.transpose(dm[next, range(next + 1)])
    true = np.append(true, [np.zeros(len(true[1, :]))], 0)
    true = np.append(true, np.transpose([np.zeros(len(true[1, :]) + 1)]), 1)

    true[:, next] = all_dtw
    true[next, :] = all_dtw
    return true


def cluster_stream(series, start_index, skip):
    full_dm = np.loadtxt('CBF_DM_nn.csv', delimiter=',')
    location = "results/CBF/error/" + str(start_index) + "/"
    fileName= location + "error.csv"

    while True:
        true = full_dm[range(start_index), :]
        true = true[:, range(start_index)]

        try:
            results_error = np.loadtxt(fileName, delimiter=',')
            results_error = np.append(results_error, [np.zeros(len(results_error[0, :]))], 0)
        except:
            results_error = np.zeros((1, int((len(series) - start_index) / skip)))

        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args)
        approx = ACA(cp, tolerance=0.05, max_rank=5000)
        index = start_index
        save_index = 0

        error = np.linalg.norm(true - approx.getApproximation(full_dm))
        results_error[len(results_error) - 1, save_index] = error
        print(len(approx.rows))
        print(error / index, len(approx.getApproximation(full_dm)))
        save_index += 1
        while index < len(series)-1:
            index += 1
            true = add_series_to_dm(true, index, full_dm)
            approx.extend(series[index], full_dm, method="v1")
            if index % skip == 0:
                print(len(approx.rows))
                error = np.linalg.norm(true - approx.getApproximation(full_dm))
                print(error/index, len(approx.getApproximation(full_dm)))
                results_error[len(results_error) - 1, save_index] = error
                save_index += 1

        np.savetxt(fileName,results_error, delimiter=',')
        if len(results_error) > 10:
            break

cluster_stream(series_tr, 400, 1)