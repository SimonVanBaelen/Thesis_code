from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.extendable_aca import ACA
import numpy as np

"""
Test code for debugging purposes
"""

names = ["CBF"]

name = names[0]
path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
labels_train, series_train = load_timeseries_from_tsv(path_train)

path_test = "Data/" + name + "/" + name + "_TEST.tsv"
labels_tr, series = load_timeseries_from_tsv(path_test)

func_name = "dtw"
args = {"window": len(series[0])-1}


full_dm = np.load('distance_matrices/CBF_DM_nn.npy')


start_index = 400
true = full_dm[range(start_index), :]
true = true[:, range(start_index)]

cp = ClusterProblem(series[0:start_index], func_name, compare_args=args)
approx1 = ACA(cp, tolerance=0.05, max_rank=20)
control_index = len(approx1.getApproximation(full_dm))-1
approx1.extend(series[start_index+1], full_dm, method="v2")
a1 = approx1.getApproximation(full_dm)
# print(len(a1), a1[400])
for row in approx1.rows:
    print(row[len(row)-1])
# indices = approx1.indices
# approx2 = ACA(cp, tolerance=0.05, max_rank=20, start_indices=indices, restart_deltas=approx1.deltas, restart_with_prev_pivots=True)
# a2 = approx2.getApproximation(full_dm)
