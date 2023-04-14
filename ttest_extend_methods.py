from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
import numpy as np

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


start_index = 500
true_start = full_dm[range(start_index), :]
true_start = true_start[:, range(start_index)]
true_extended = full_dm[range(start_index), :]
true_extended = true_extended[:, range(start_index)]
cp_start = ClusterProblem(series[0:start_index], func_name, compare_args=args)

for index in range(start_index+1, start_index+25):
    cp_extended = ClusterProblem(series[0:index], func_name, compare_args=args)
    approx_start = ACA(cp_start, tolerance=0.05, max_rank=20, seed=255, start_index=255)
    approx_start.extend(series[index], full_dm, method="v4")
    approx_extended = ACA(cp_extended, tolerance=0.05, max_rank=20, seed=255, start_index=255, restart_deltas=2, start_samples = [approx_start.sample_indices, approx_start.sample_values])
    if index == 502:
        for i in range(len(approx_start.rows)):
            print("index extended:", index, i, approx_start.indices)
            print("index start:", index, i, approx_extended.indices)
            # print("index extended:", index, i, approx_extended.rows[14][approx_start.indices[15]])
            # print("index start:", index, i, approx_start.rows[14][approx_start.indices[15]])
            # print("row elements:", index, i, np.array_equal(approx_start.rows[i], approx_extended.rows[i]))
            # print("deltas:", i, np.array_equal(approx_start.rows[i], approx_extended.rows[i]))
        break

