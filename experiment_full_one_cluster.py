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

def calculateClusters(approx, index, labels, cluster_algo,k):
    temp_approx = np.exp(- approx ** 2 / 4.529)

    model_spec = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0)
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
    for file_name in file_names:
        try:
            result = np.load(file_name + ".npy")
            n_skips = int((size - start_index) / skip)
            results.append(np.append(result, [np.zeros((5, n_skips))], 0))
        except: # TODO add type of exception
            results.append(np.zeros((1, 5, int((size - start_index) / skip))))
    return results


def do_full_experiment(series, labels, dm, start_index, skip, methods, cluster_algo, rank=15, iterations=100, random_file=True):
    func_name = "dtw"
    args = {"window": len(series) - 1}
    k = len(set(labels))
    file_names = []
    seed_file_name = rn.randint(0,9999999999)
    for method in methods:
        if random_file:
            file_names.append("results/CBF_full" + "_" + str(seed_file_name) + "_" + method + "_spectral_unlimited_rank")
        else:
            file_names.append("results/CBF_full" + "_" + method + "_spectral_unlimited_rank")
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
            approximations.append(ACA(copy.deepcopy(cp), adaptive=methods[i]=="method3", tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed))

        index = start_index
        update_results(approximations, results, labels, active_dm, cluster_algo, k, index, start_index, skip)
        while index < len(series) - 1:
            index += 1
            print(index)
            if dm is not None:
                active_dm = dm[range(index), :]
                active_dm = active_dm[:, range(index)]
            new_serie = series[index]
            extend_approximations(approximations, methods, new_serie, solved_matrix=active_dm)
            if index % skip == 0:
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

def modify_data(series, labels, true_dm, modify_name=None):
    if modify_name == 'stagnate':
        pass
    elif modify_name == 'noise':
        pass
    elif modify_name == 'unknown_label':
        pass
    return series, labels, true_dm

name = "CBF"
series, labels = load_data(name)
true_dm = np.loadtxt("distance_matrices/"+name+'_DM_nn.csv', delimiter=',')
series, labels, true_dm = modify_data(series, labels, true_dm)
methods = ["method1", "method2", "method3", "method4", "method5"]
start = int(len(series)/2)
skip = int((len(series)-start)/15)
print("start: ", start,"Skip: ", skip)
do_full_experiment(series, labels, true_dm, start, skip, methods, "spectral", rank=9000, iterations=1000)