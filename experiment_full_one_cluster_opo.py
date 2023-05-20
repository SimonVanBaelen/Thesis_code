import copy
import sys

import pandas as pd
from sklearn.cluster import SpectralClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score
from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
import random as rn
import make_new_datasets
from dtaidistance import dtw, clustering
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

def calculateClusters(approx, index, labels, cluster_algo,k):
    temp_approx = np.exp(- approx ** 2 / 4.021)

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


def extend_approximations(approximations, methods, new_serie, solved_matrix=None):
    for approximation, method in zip(approximations, methods):
        approximation.extend(new_serie, method=method, solved_matrix=solved_matrix)


def print_result(new_result):
    # print("relative_error ", relative_error)
    # print("Spectral", "Iteration " + str(index), "DTW ARI:", ARI_scoreDTW)
    print("Spectral", "Iteration", new_result[0], "Approx ARI:", new_result[3])


def update_results(approximations, results, labels, true_dm, cluster_algo, k, index, start_index, skip):
    for approx, result in zip(approximations, results):
        ARI_score = calculateClusters(approx.getApproximation(), index, labels, cluster_algo, k)
        relative_error = np.sqrt(np.average(np.square(true_dm - approx.getApproximation())))
        amount_of_skeletons = len(approx.rows) + len(approx.full_dtw_rows)
        amount_of_dtws = approx.get_DTW_calculations()
        new_result = [index, amount_of_skeletons, relative_error, ARI_score, amount_of_dtws]
        print_result(new_result)
        result[len(result) - 1, :, int((index - start_index) / skip)] = np.array(new_result)


def read_all_results(file_names, size, start_index, skip):
    results = []
    n_skips = int((size - start_index) / skip) + 1
    for file_name in file_names:
        try:
            result = np.load(file_name + ".npy")
            results.append(np.append(result, [np.zeros((5, n_skips))], 0))
        except:
            results.append(np.zeros((1, 5, n_skips)))
    return results


def do_full_experiment(series, labels, dm, start_index, skip, methods, cluster_algo, rank=15, iterations=100, random_file=True):
    func_name = "dtw"
    args = {"window": len(series) - 1}
    k = len(set(labels))
    file_names = []
    seed_file_name = rn.randint(0,9999999999)
    for method in methods:
        if random_file:
            file_names.append("results/stagnate/" + str(seed_file_name) + "_" + method + "_full")
        else:
            file_names.append("results/stagnate/" + method + "_full")
    while True:
        if dm is not None:
            active_dm = dm[range(start_index), :]
            active_dm = active_dm[:, range(start_index)]
        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args, solved_matrix=active_dm)
        results = read_all_results(file_names, len(series), start_index, skip)
        start_index_approx = rn.randint(0,start_index-1)
        seed = rn.randint(0,99999999)
        print("STARTING NEW APPROX: it =", len(results[0]), "start index approx =", start_index_approx, "seed =", seed)
        approximations = [ACA(cp, tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed)]
        for i in range(1, len(methods)):
            approximations.append(ACA(copy.deepcopy(cp), tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed))

        index = start_index
        update_results(approximations, results, labels, active_dm, cluster_algo, k, index, start_index, skip)
        while index < len(series) - 1:
            index += 1
            if dm is not None:
                active_dm = dm[range(index), :]
                active_dm = active_dm[:, range(index)]
            new_serie = series[index]
            extend_approximations(approximations, methods, [new_serie], solved_matrix=active_dm)
            if index % skip == 0:
                update_results(approximations, results, labels, active_dm, cluster_algo, k, index, start_index, skip)
        update_results(approximations, results, labels, active_dm, cluster_algo, k, index, start_index, skip)
        for file_name, result in zip(file_names, results):
            np.save(file_name, result)

        if len(results[0]) > iterations:
            break


def load_data(name):
    path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
    labels_train, series_train = load_timeseries_from_tsv(path_train)

    path_test = "Data/" + name + "/" + name + "_TEST.tsv"
    labels_test, series_test = load_timeseries_from_tsv(path_test)

    labels = np.concatenate((labels_train, labels_test), axis=0)
    series = np.concatenate((series_train, series_test), axis=0)
    return series, labels


name = "CBF"
series, labels = load_data(name)
true_dm = np.loadtxt("distance_matrices/"+name+'_DM_nn.csv', delimiter=',')
series, labels, true_dm = make_new_datasets.modify_data(series, labels, true_dm,'stagnate')
methods = ["method1", "method2", "method3", "method4", "method5"]
start = 400
skip = int((len(series)-start)/10)
print("start: ", start,"Skip: ", skip)
do_full_experiment(series, labels, true_dm, start, skip, methods, "spectral", rank=9000, iterations=1000)