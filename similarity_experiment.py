import math
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

# from plot_properties_dataset import get_distance_matrix_between_labels
from src.aca import ACA
from src.cluster_problem import ClusterProblem
import numpy as np

from src.data_loader import load_timeseries_from_tsv

def get_distance_matrix_between_labels(l1, l2, labels, dm):
    indices_l1 = np.where(labels == l1)[0]
    indices_l2 = np.where(labels == l2)[0]
    tmp = dm[indices_l1, :]
    return tmp[:, indices_l2]
# "FiftyWords", "CBF", "FaceAll", "StarLightCurves"
names = ["FiftyWords", "CBF", "FaceAll", "StarLightCurves", "ECG5000", "ElectricDevices", "EthanolLevel", "ChlorineConcentration", "Wafer", "Crop"]
for name in names:
    print(name)
    path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
    labels_train, series_train = load_timeseries_from_tsv(path_train)

    path_test = "Data/" + name + "/" + name + "_TEST.tsv"
    labels_test, series_test = load_timeseries_from_tsv(path_test)

    labels = np.concatenate((labels_train, labels_test), axis=0)
    series = np.concatenate((series_train, series_test), axis=0)


    func_name = "dtw"         # "msm"/"ed" other options
    args = {"window": len(series[0])-1}   # for MSM "c" parameter
    start = int(len(series)/2)
    series = series[0:start]
    labels = labels[0:start]
    cp = ClusterProblem(series, func_name, compare_args=args)
    distance_matrix = ACA(cp, tolerance=0.05, max_rank=9000).getApproximation()

    same = []
    diff = []
    for l1 in labels:
        for l2 in labels:
            dm_between_labels = get_distance_matrix_between_labels(l1, l2, labels, distance_matrix)
            if l1 == l2:
                same.append(np.average(dm_between_labels))
            else:
                diff.append(np.average(dm_between_labels))

    print("Same label:", min(same), max(same), np.average(same), np.median(same))
    print("Diff label:", min(diff), max(diff), np.average(diff), np.median(diff))

    def do_clustering_sim_diff_functions(distance_matrix, labels, functions, possible_elements):
        model_spec = SpectralClustering(n_clusters=len(set(labels)), affinity='precomputed', assign_labels='discretize',random_state=0)
        for f in functions:
            ari_for_function = []
            for e in possible_elements:
                similarity_matrix = f(distance_matrix, e)
                prediction = model_spec.fit_predict(similarity_matrix)
                ari_for_function.append(adjusted_rand_score(labels,prediction))
            print(ari_for_function)

    def function1(approx, e):
        lowest = 0.01
        a = (-e ** 2) / math.log(lowest)
        return np.exp(- (approx ** 2) / a)

    def function2(approx, e):
        return -np.tanh(500*(approx-e))/2+0.5

    def function3(approx, e):
        return np.maximum(0, np.exp(-approx / e + 1))

    possible_elements = [max(same), np.average(same), min(diff), np.average(diff)]
    functions = [function1, function2, function3]
    do_clustering_sim_diff_functions(distance_matrix, labels, functions, possible_elements)