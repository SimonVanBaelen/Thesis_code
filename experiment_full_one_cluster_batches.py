import copy
import sys

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
import random as rn
from dtaidistance import dtw, clustering
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

def calculateClusters(approx, index, labels, k):
    temp_approx = np.exp(- approx ** 2 / 8)

    model_spec = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=0)
    result_spec = model_spec.fit_predict(temp_approx)
    norm_Approx_spectral = adjusted_rand_score(labels[0:index], result_spec)

    return norm_Approx_spectral


def add_series_to_dm(true, next, dm):
    next = next - 1
    all_dtw = np.transpose(dm[next, range(next + 1)])
    true = np.append(true, [np.zeros(len(true[1, :]))], 0)
    true = np.append(true, np.transpose([np.zeros(len(true[1, :]) + 1)]), 1)

    true[:, next] = all_dtw
    true[next, :] = all_dtw
    return true


def rearrange_data(labels, series, full_dm):
    indices1 = np.where(labels == 1)[0][range(0, 250)]
    indices2 = np.where(labels == 2)[0][range(0, 25)]
    indices12 = np.concatenate((indices1, indices2))
    np.random.shuffle(indices12)
    indices3 = np.where(labels == 1)[0][range(250, 265)]
    all_indices = np.concatenate((indices12, indices3))

    new_series = series[all_indices]
    new_labels = labels[all_indices]
    new_dm = full_dm[all_indices, :]
    new_dm = new_dm[:, all_indices]

    return new_labels, new_series, new_dm


def extend_approximations(approximations, methods, new_serie):
    for approximation, method in zip(approximations, methods):
        approximation.extend(new_serie, method=method)


def print_result(new_result):
    # print("relative_error ", relative_error)
    # print("Spectral", "Iteration " + str(index), "DTW ARI:", ARI_scoreDTW)
    print("Spectral", "Iteration", new_result[0], "Approx ARI:", new_result[3])


def update_results(approximations, results, labels, k, index, start_index, skip):
    for approx, result in zip(approximations, results):
        ARI_score = calculateClusters(approx.getApproximation(), index, labels, k)
        # relative_error = np.sqrt(np.average(np.square(true - approx.getApproximation(full_dm)))) #TODO
        amount_of_skeletons = len(approx.rows) + len(approx.full_dtw_rows)
        amount_of_dtws = approx.get_DTW_calculations()
        relative_error = 0
        new_result = [index, amount_of_skeletons, relative_error, ARI_score, amount_of_dtws]
        print_result(new_result)
        result[len(result) - 1, :, int((index - start_index) / skip)] = np.array(new_result)


def read_all_results(file_names, size, start_index, skip):
    results = []
    for file_name in file_names:
        try:
            result = np.load(file_name + ".npy")
            n_skips = int((size - start_index) / skip)
            results.append(np.append(result, [np.zeros((5, n_skips))], 0))
        except: # TODO add type of exception
            results.append(np.zeros((1, 5, int((size - start_index) / skip))))
    return results


def do_full_experiment(labels, series, start_index, skip, methods, rank=15, iterations=100):
    # full_dm = np.loadtxt('CBF_DM_nn.csv', delimiter=',')
    # labels, series, full_dm = rearrange_data(labels, series, full_dm)
    k = len(set(labels))
    file_names = []
    for method in methods:
        file_names.append("results/CBF_full" + "_" + method + "_spectral_unlimited_rank")
    while True:
        # true = full_dm[range(start_index), :]
        # true = true[:, range(start_index)]
        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args)
        results = read_all_results(file_names, len(series), start_index, skip)

        # TODO add seed + start index + print them
        start_index_approx = rn.randint(0,start_index-1)
        seed = rn.randint(0,99999999)
        print("STARTING NEW APPROX: it =", len(results[0]), "start index approx =", start_index_approx, "seed =", seed)
        approximations = [ACA(cp, tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed)]
        for i in range(1, len(methods)):
            approximations.append(copy.deepcopy(approximations[0]))

        index = start_index
        update_results(approximations, results, labels, k, index, start_index, skip)
        while index < len(series) - 1:
            index += 1
            print(len(results[0]), index)
            new_serie = series[index]
            extend_approximations(approximations, methods, new_serie)
            if index % skip == 0:
                update_results(approximations, results, labels, k, index, start_index, skip)

        for file_name, result in zip(file_names, results):
            np.save(file_name, result)

        if len(results[0]) > iterations:
            break



name = "CBF" # sys.argv[1]
path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
labels_train, series_train = load_timeseries_from_tsv(path_train)

path_test = "Data/" + name + "/" + name + "_TEST.tsv"
labels_test, series_test = load_timeseries_from_tsv(path_test)

labels = np.concatenate((labels_train, labels_test), axis=0)
series = np.concatenate((series_train, series_test), axis=0)

func_name = "dtw"  # "msm"/"ed" other options
args = {"window": len(series) - 1}  # for MSM "c" parameter

methods = ["method1", "method2", "method3", "method4", "method5"]
start = int(len(series)/2)
skip = int(start/15)
# skip = 25
print("start: ", start,"Skip: ", skip)
do_full_experiment(labels, series, start, skip, methods, rank=9000, iterations=1000)

# cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=9000, iterations=100, method="method2")
# cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=9000, iterations=100, method="method3")
# cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=9000, iterations=100, method="method4")
# cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=20, iterations=100, method="method4")
# cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method1")
# cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method2")
# cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method3")
# cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method4")

# model = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
# cluster_idx = model.fit(series_tr[0:9])
# model.plot("hierarchy.png")