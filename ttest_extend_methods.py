from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
import numpy as np
import random as rnd

# FacesUCR, FaceAll
names = ["CBF"]

name = names[0]
path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
labels_train, series_train = load_timeseries_from_tsv(path_train)

path_test = "Data/" + name + "/" + name + "_TEST.tsv"
labels_tr, series = load_timeseries_from_tsv(path_test)

func_name = "dtw"         # "msm"/"ed" other options
args = {"window": len(series[0])-1}   # for MSM "c" parameter


full_dm = np.loadtxt('CBF_DM_nn.csv', delimiter=',')


start_index = 400
true_start = full_dm[range(start_index), :]
true_start = true_start[:, range(start_index)]
true_extended = full_dm[range(start_index), :]
true_extended = true_extended[:, range(start_index)]
# start index = 114 seed = 74241553
# 255
for _ in range(0, 100):
    start_i = 341
        # rnd.randint(0, start_index - 1)
    seed = 83831336
        # rnd.randint(0, 100000000000)
    cp_start = ClusterProblem(series[0:start_index], func_name, compare_args=args, solved_matrix=true_start)
    approx_start = ACA(cp_start, tolerance=0.05, max_rank=9000, seed=seed, start_index=start_i)
    print("random start: ", start_i)
    print("random seed: ", seed)
    for index in range(start_index+1, start_index+300):
        true = full_dm[range(index), :]
        true = true[:, range(index)]
        tmp = series[0:index]
        cp_extended = ClusterProblem(tmp, func_name, compare_args=args, solved_matrix=true)
        approx_start.extend(series[index-1], true, method="method3")
        approx_extended = ACA(cp_extended, tolerance=0.05, max_rank=9000, seed=seed, start_index=start_i, restart_deltas=2, start_samples=[approx_start.sample_indices, approx_start.sample_values])
        if index == 419:
            for i in range(len(approx_start.rows)):
                print(index, i, np.array_equal(approx_extended.rows[i], approx_start.rows[i]))
            print(index, np.array_equal(approx_extended.deltas, approx_start.deltas))
            print(len(approx_extended.rows), len(approx_start.rows))
            print(approx_start.current_rank, approx_extended.current_rank)
            print(approx_extended.getApproximation(true)- approx_start.getApproximation(true))
            break
    break
        # print(approx_start.indices, approx_extended.indices)
        # print(approx_start.rows[0][-1] - approx_extended.rows[0][-1])
        # for i in range(len(approx_start.rows)):
        #     print(index, i, np.array_equal(approx_extended.rows[i], approx_start.rows[i]))


