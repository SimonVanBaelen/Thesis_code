import copy
import sys

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

import make_new_datasets
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

def print_result(new_result):
    # print("relative_error ", relative_error)
    # print("Spectral", "Iteration " + str(index), "DTW ARI:", ARI_scoreDTW)
    print("Spectral", "Iteration", new_result[0], "Approx ARI:", new_result[1])


def update_results(result, labels, active_dm, index, start_index, skip):
    ARI_score = calculateClusters(active_dm, index, labels, "none", k=len(set(labels)))
    amount_of_dtws = index*index/2 - index
    print(index*index/2 - index)
    new_result = [index, ARI_score, amount_of_dtws]
    print_result(new_result)
    result[len(result) - 1, :, int((index - start_index) / skip)] = np.array(new_result)


def read_all_results(size, start_index, skip):
    return np.zeros((1, 3, int((size - start_index) / skip)+1))


def do_full_experiment(series, labels, dm, start_index, skip, name):
    file_name = "full_dm_ari/" + name + "_full_dm_results_"+str(skip)+"_"+str(start_index)
    active_dm = dm[range(start_index), :]
    active_dm = active_dm[:, range(start_index)]

    results = read_all_results(len(series), start_index, skip)

    index = start_index
    update_results(results, labels, active_dm, index, start_index, skip)
    while index < len(series) - 1:
        index += 1
        if dm is not None:
            active_dm = dm[range(index), :]
            active_dm = active_dm[:, range(index)]

        if index % skip == 0:
            update_results(results, labels, active_dm, index, start_index, skip)
    update_results(results, labels, active_dm, index, start_index, skip)
    np.save(file_name, results)


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
series, labels, true_dm = make_new_datasets.modify_data(series, labels, true_dm)
start = int(len(series)/2)
skip = 50
print(skip)
print("start: ", start,"Skip: ", skip)
do_full_experiment(series, labels, true_dm, start, skip, name)