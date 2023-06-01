import copy
import os
import sys
import os

path = os.getcwd()

print(path)
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
import random as rn
import numpy as np



def calculateClusters(approx, index, labels, a_spectral, k, name):
    # if a_spectral is None:
        # a_spectral = calculate_spectral_var(approx, labels[0:index], name)
    temp_approx = np.exp(- approx ** 2 / a_spectral)

    spectral = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=0)
    agglomerative = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
    pred_spectral = spectral.fit_predict(temp_approx)
    pred_agglo = agglomerative.fit_predict(approx)
    ARI_Approx_spectral = adjusted_rand_score(labels[0:index], pred_spectral)
    ARI_Approx_agglomerative = adjusted_rand_score(labels[0:index], pred_agglo)
    print(ARI_Approx_spectral, ARI_Approx_agglomerative)
    return ARI_Approx_spectral, ARI_Approx_agglomerative
    # return 0, 0

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


def update_results(approximations, results, labels, true_dm, a_spectral, k, index, start_index, skip, name):
    for approx, result in zip(approximations, results):
        ARI_score_spec_approx, ARI_score_agglo_approx = calculateClusters(approx.getApproximation(), index, labels,
                                                                          a_spectral, k, name)
        ARI_score_spec_exact, ARI_score_agglo_exact = calculateClusters(true_dm, index, labels, a_spectral, k, name)
        amount_of_skeletons = len(approx.rows) + len(approx.full_dtw_rows)
        amount_of_dtws = approx.get_DTW_calculations()
        new_result = [index, amount_of_skeletons, ARI_score_spec_exact, ARI_score_agglo_exact,
                      ARI_score_spec_approx, ARI_score_agglo_approx, amount_of_dtws]
        print_result(new_result)
        result[len(result) - 1, :, int((index - start_index) / skip)] = np.array(new_result)


def read_all_results(file_names, size, start_index, skip):
    results = []
    for file_name in file_names:
        try:
            result = np.load(file_name + ".npy")
            n_skips = int((size - start_index) / skip)
            results.append(np.append(result, [np.zeros((7, n_skips))], 0))
        except:
            results.append(np.zeros((1, 7, int((size - start_index) / skip))))
    return results


def do_full_experiment(series, labels, dm, start_index, skip, methods, cluster_algo, a_spectral, name, rank=15,
                       iterations=100, random_file=True):
    func_name = "dtw"
    args = {"window": len(series) - 1}
    k = len(set(labels))
    file_names = []
    seed_file_name = rn.randint(0, 9999999999)
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
        start_index_approx = rn.randint(0, start_index - 1)
        seed = rn.randint(0, 99999999)
        print(name + ":" + " STARTING NEW APPROX: it =", len(results[0]), "start index approx =", start_index_approx,
              "seed =", seed, "skip =", skip)
        approximations = [ACA(cp, tolerance=0.05, max_rank=rank, start_index=start_index_approx, seed=seed)]
        index = start_index
        update_results(approximations, results, labels, active_dm, a_spectral, k, index, start_index, skip, name)
        new_series = []
        while index < len(series) - 1:
            index += 1
            new_series.append(series[index])
            if index % skip == 0:
                if dm is not None:
                    active_dm = dm[range(index), :]
                    active_dm = active_dm[:, range(index)]
                else:
                    active_dm = None
                extend_approximations(approximations, methods, new_series, solved_matrix=active_dm)
                update_results(approximations, results, labels, active_dm, a_spectral, k, index, start_index, skip,
                               name)
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


# already_found_names = ["CBF", "ChlorineConcentration", "CinCECGTorso"]
# names = [x[1] for x in [y[0].split("\\") for y in os.walk("Data")][1:-1] if x[1] not in already_found_names]
names = ["Wafer"]
variables_spectral = [8.691]
for name, a_spectral in zip(names, variables_spectral):
    series, labels = load_data(name)
    true_dm = np.load("distance_matrices/" + name + '_DM_nn.npy')
    methods = ["method3"]
    start = int(len(series) / 2)
    skip = int((len(series) - start) / 10)
    if name == "Wafer":
        skip = int((len(series) - start) / 5)
    print("start: ", start, "Skip: ", skip)

    do_full_experiment(series, labels, true_dm, start, skip, methods, "spectral", a_spectral, name, rank=9000,
                       iterations=1000, random_file=False)