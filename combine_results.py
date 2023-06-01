from os import listdir
from os.path import isfile, join
import numpy as np

path_load = 'results/CBF/seperate_full/new/'
path_save = 'results/part1/new_label/'
file_names = [f for f in listdir(path_load) if isfile(join(path_load, f))]

method_results = [None, None, None, None, None]
print(len(file_names))
for fn in file_names:
    print(fn.split("_"))
    index = int(fn.split("_")[1][-1])-1
    print(fn.split("_"))
    new_results = np.load(path_load+fn)
    if method_results[index] is None:
        method_results[index] = new_results
    else:
        method_results[index] = np.append(method_results[index], new_results, 0)

names = ["method1_full.npy", "method2_full.npy", "method3_full.npy", "method4_full.npy", "method5_full.npy"]
for i in range(len(method_results)):
    np.save(path_save+names[i], method_results[i])