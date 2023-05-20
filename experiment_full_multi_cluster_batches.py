import copy
import os
import sys

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from similarity_experiment import get_distance_matrix_between_labels
from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
import random as rn
from dtaidistance import dtw, clustering
import scipy.stats as stats
import matplotlib.pyplot as plt
from statistics import mean
import numpy as np
from numpy.linalg import norm


def calculate_spectral_var(approx, labels, name):
    same = []
    for l1 in set(labels):
        for l2 in set(labels):
            dm_between_labels = get_distance_matrix_between_labels(l1, l2, labels, approx)
            if l1 == l2:
                for v in dm_between_labels:
                        same.append(np.average(v))
    filename = "Data/" + name +"/" + name +"_spec_v.txt"
    mean = np.average(same)
    np.savetxt(filename, np.zeros(1)+mean, delimiter=",")
    return mean



def calculateClusters(approx, index, labels, cluster_algo, a_spectral, k, name):
    if a_spectral is None:
        a_spectral = calculate_spectral_var(approx, labels[0:index], name)
    temp_approx = np.exp(- approx ** 2 / a_spectral)

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


def extend_approximations(approximations, methods, new_series, solved_matrix=None):
    for approximation, method in zip(approximations, methods):
        approximation.extend(new_series, method=method, solved_matrix=solved_matrix)


def print_result(new_result):
    print("Spectral", "Iteration", new_result[0], "Approx ARI:", new_result[3])


def update_results(approximations, results, labels, true_dm, cluster_algo, a_spectral, k, index, start_index, skip, name):
    for approx, result in zip(approximations, results):
        ARI_score = calculateClusters(approx.getApproximation(), index, labels, cluster_algo, a_spectral, k, name)
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
        except:
            results.append(np.zeros((1, 5, int((size - start_index) / skip))))
    return results


def do_full_experiment(series, labels, dm, start_index, skip, methods, cluster_algo, a_spectral, name, rank=15, iterations=100, random_file=True):
    func_name = "dtw"
    args = {"window": len(series) - 1}
    k = len(set(labels))
    file_names = []
    seed_file_name = rn.randint(0,9999999999)
    for method in methods:
        if random_file:
            file_names.append("results/part2/" + name + "/" + str(seed_file_name) + "_" + method)
        else:
            file_names.append("results/part2/" + name + "/" + name + "_" + method)
    results = read_all_results(file_names, len(series), start_index, skip)
    while len(results[0]) <= iterations:
        if dm is not None:
            active_dm = dm[range(start_index), :]
            active_dm = active_dm[:, range(start_index)]
        else:
            active_dm = None
        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args, solved_matrix=active_dm)
        results = read_all_results(file_names, len(series), start_index, skip)
        start_index_approx = rn.randint(0,start_index-1)
        seed = rn.randint(0,99999999)
        print(name + ":" + " STARTING NEW APPROX: it =", len(results[0]), "start index approx =", start_index_approx, "seed =", seed, "skip =", skip)
        approximations = [ACA(cp, tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed)]
        for i in range(1, len(methods)):
            approximations.append(ACA(copy.deepcopy(cp), tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed))

        index = start_index
        update_results(approximations, results, labels, active_dm, cluster_algo, a_spectral, k, index, start_index, skip, name)
        new_series = []
        while index < len(series) - 1:
            index += 1
            new_series.append(series[index])
            if dm is not None:
                active_dm = dm[range(index), :]
                active_dm = active_dm[:, range(index)]
            else:
                active_dm = None
            if index % skip == 0:
                extend_approximations(approximations, methods, new_series, solved_matrix=active_dm)
                update_results(approximations, results, labels, active_dm, cluster_algo, a_spectral, k, index, start_index, skip, name)
                new_series = []

        for file_name, result in zip(file_names, results):
            np.save(file_name, result)


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

already_found_names = ["CBF", "ChlorineConcentration", "CinCECGTorso"]
names = [x[1] for x in [y[0].split("\\") for y in os.walk("Data")][1:-1] if x[1] not in already_found_names]
for name in names:
    series, labels = load_data(name)
    true_dm = np.loadtxt("distance_matrices/"+name+'_DM_nn.csv', delimiter=',')
    series, labels, true_dm = modify_data(series, labels, true_dm)
    methods = ["method1", "method3"]
    start = int(len(series)/2)
    skip = int((len(series)-start)/10)
    print("start: ", start,"Skip: ", skip)


    def read_value_for_spectral(name):
        try:
            filename = "Data/" + name +"/" + name +"_spec_v.txt"
            value = np.loadtxt(filename)
        except:
            value = None
        return value

    a_spectral = read_value_for_spectral(name)
    do_full_experiment(series, labels, None, start, skip, methods, "spectral", a_spectral, name, rank=9000, iterations=1, random_file=False)