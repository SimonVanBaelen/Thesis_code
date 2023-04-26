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
print(len(series_tr[10]), len(series_tr))
print(len(series_train[10]), len(series_train))
# series_tr = np.concatenate((series_tr, series_train))
# labels_tr = np.concatenate((labels_tr, labels_train))

func_name = "dtw"         # "msm"/"ed" other options
args = {"window": len(series_tr[0])-1}   # for MSM "c" parameter


def calculateClusters(true, approx, index, labels, k):
    temp_true = np.exp(- true ** 2 / (2 ** 2))
    temp_approx = np.exp(- approx ** 2 / (2 ** 2))
    # temp_true = true
    # temp_approx = approx

    model_spec = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0)
    result_spec = model_spec.fit_predict(temp_true)
    norm_DTW_spectral = adjusted_rand_score(labels[0:index], result_spec)

    model_agglo = AgglomerativeClustering(n_clusters=k, linkage='complete', affinity='precomputed')
    result_agglo = model_agglo.fit_predict(temp_true)
    norm_DTW_agglo = adjusted_rand_score(labels[0:index], result_agglo)
    cm2 = confusion_matrix(labels[0:index], result_agglo)
    print(len(set(result_agglo)), cm2)

    model_spec = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0)
    result_spec = model_spec.fit_predict(temp_approx)
    norm_Approx_spectral = adjusted_rand_score(labels[0:index], result_spec)

    model_agglo = AgglomerativeClustering(n_clusters=k, linkage='complete', affinity='precomputed')
    result_agglo = model_agglo.fit_predict(temp_approx)
    norm_Approx_agglo = adjusted_rand_score(labels[0:index], result_agglo)

    return norm_DTW_spectral, norm_Approx_spectral, norm_DTW_agglo, norm_Approx_agglo


def saveNorm(normDTW, normApprox, results, index):
    results[0,index] = normDTW
    results[len(results)-1, index] = normApprox
    return index+1

def add_series_to_dm(true, next, dm):
    next = next - 1
    all_dtw = np.transpose(dm[next, range(next + 1)])
    true = np.append(true, [np.zeros(len(true[1, :]))], 0)
    true = np.append(true, np.transpose([np.zeros(len(true[1, :]) + 1)]), 1)

    true[:, next] = all_dtw
    true[next, :] = all_dtw
    return true


def rearrange_data(labels, series, full_dm):
    indices1 = np.where(labels == 1)[0][range(0,250)]
    indices2 = np.where(labels == 2)[0][range(0,25)]
    indices12 = np.concatenate((indices1,indices2))
    np.random.shuffle(indices12)
    indices3 = np.where(labels == 1)[0][range(250,265)]
    all_indices = np.concatenate((indices12,indices3))

    new_series = series[all_indices]
    new_labels = labels[all_indices]
    new_dm = full_dm[all_indices, :]
    new_dm = new_dm[:, all_indices]

    return new_labels, new_series, new_dm


def cluster_stream(labels, series, start_index, skip, method, rank=15, iterations=100):
    full_dm = np.loadtxt('CBF_DM_nn.csv', delimiter=',')
    # labels, series, full_dm = rearrange_data(labels, series, full_dm)
    k = len(set(labels))
    location = "results/CBF/clustering/" + str(start_index) + "/" + method + "/"
    file_Name_spec = location + "full_dtw_ari_800.csv"
    file_Name_agglo = location + "agglo.csv"

    while True:
        true = full_dm[range(start_index), :]
        true = true[:, range(start_index)]
        try:
            results_ari_spec = np.loadtxt(file_Name_spec, delimiter=',')
            results_ari_spec = np.append(results_ari_spec, [np.zeros(len(results_ari_spec[0, :]))], 0)

            results_ari_agglo = np.loadtxt(file_Name_agglo, delimiter=',')
            results_ari_agglo = np.append(results_ari_agglo, [np.zeros(len(results_ari_agglo[0, :]))], 0)
        except:
            results_ari_spec = np.zeros((2, int((len(series) - start_index)/skip)))
            results_ari_agglo = np.zeros((2, int((len(series) - start_index) / skip)))

        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args)
        approx = ACA(cp, tolerance=0.05, max_rank=rank)

        index = start_index
        save_index = 0
        normDTWSpec, normApproxSpec, normDTWAgglo, normApproxAgglo = calculateClusters(true, approx.getApproximation(full_dm), index, labels, k)
        save_index = saveNorm(normDTWSpec, normApproxSpec, results_ari_spec, save_index)
        saveNorm(normDTWAgglo, normApproxAgglo, results_ari_agglo, save_index)
        print("Spectral", "Iteration " + str(index), "DTW ARI:", normDTWSpec)
        print("Spectral", "Iteration " + str(index), "Approx ARI:", normApproxSpec)
        # print("Agglomerative", "Iteration " + str(index), "DTW ARI:", normDTWAgglo)
        # print("Agglomerative", "Iteration " + str(index), "Approx ARI:", normApproxAgglo)
        while index < len(series)-1:
            index += 1
            true = add_series_to_dm(true, index, full_dm)
            approx.extend(series[index], full_dm, method=method)
            if index % skip == 0:
                normDTWSpec, normApproxSpec, normDTWAgglo, normApproxAgglo = calculateClusters(true, approx.getApproximation(full_dm), index, labels, k)
                saveNorm(normDTWSpec, normApproxSpec, results_ari_spec, save_index)
                save_index = saveNorm(normDTWAgglo, normApproxAgglo, results_ari_agglo, save_index)
                print("Spectral", "Iteration " + str(index), "DTW ARI:", normDTWSpec)
                print("Spectral", "Iteration " + str(index), "Approx ARI:", normApproxSpec)
                print(len(approx.rows))
                # print("Agglomerative", "Iteration " + str(index), "DTW ARI:", normDTWAgglo)
                # print("Agglomerative", "Iteration " + str(index), "Approx ARI:", normApproxAgglo)
        np.savetxt(file_Name_spec,results_ari_spec, delimiter=',')
        np.savetxt(file_Name_agglo, results_ari_agglo, delimiter=',')
        if len(results_ari_spec) > iterations:
            break

cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=20, iterations=100, method="method1")
cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=20, iterations=100, method="method2")
cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=20, iterations=100, method="method3")
cluster_stream(labels_tr, series_tr, start_index=400, skip=25, rank=20, iterations=100, method="method4")

cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method1")
cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method2")
cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method3")
cluster_stream(labels_tr, series_tr, start_index=800, skip=10, rank=40, iterations=100, method="method4")