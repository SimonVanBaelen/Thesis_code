import math

import pandas as pd
from dtaidistance import dtw
from scipy.stats import stats
import make_new_datasets
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from src.aca import ACA
from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


name = "CBF" # sys.argv[1]
path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
labels_train, series_train = load_timeseries_from_tsv(path_train)

path_test = "Data/" + name + "/" + name + "_TEST.tsv"
labels_test, series_test = load_timeseries_from_tsv(path_test)

labels = np.concatenate((labels_train, labels_test), axis=0)
series = np.concatenate((series_train, series_test), axis=0)


func_name = "dtw"         # "msm"/"ed" other options
args = {"window": len(series[0])-1}   # for MSM "c" parameter


def get_distance_matrix_between_labels(l1, l2, labels, dm):
    indices_l1 = np.where(labels == l1)[0]
    indices_l2 = np.where(labels == l2)[0]
    tmp = dm[indices_l1, :]
    return tmp[:, indices_l2]

def plot_distribution_distance_of_labels(labels, distance_matrix):
    all_labels = set(labels)
    amount_of_labels = len(set(labels))
    fig, axes = plt.subplots(amount_of_labels, amount_of_labels)
    fig.suptitle('Alle afstanden tussen de verschillende labels')
    for (l1, i) in zip(all_labels, range(amount_of_labels)):
        for (l2, j) in zip(all_labels, range(amount_of_labels)):
            dm_between_labels = get_distance_matrix_between_labels(l1, l2, labels, distance_matrix)
            sns.histplot(ax=axes[i, j], data=dm_between_labels.flatten())
            axes[i, j].axvline(dm_between_labels.flatten().mean(), c='r', ls='-', lw=1.5)
            axes[i, j].set_title("Afstand tussen label " + str(int(l1)) + " en label " + str(int(l2)))
            axes[i, j].set_xlim([2, 10])
            axes[i, j].set_ylim([0, 4500])
            axes[i, j].set(xlabel="", ylabel="")
            if i == j:
                axes[i, j].set_facecolor("powderblue")
    fig.supxlabel('Afstand tussen tijdsreeksen')
    fig.supylabel('Aantal tijdsreeksen')
    plt.show()
    plt.cla()


def plot_evolution_of_distribution_of_labels(labels):
    all_labels = list(set(labels))
    amount_of_labels = len(all_labels)
    occurences = np.zeros((len(labels), amount_of_labels))
    for i in range(len(labels)):
        index = int(labels[i]-1)
        occurences[range(i, len(labels)),index] += 1
    occurences = occurences.transpose()
    for label in all_labels:
        if not label == 4:
            plt.plot(range(len(labels)), occurences[int(label-1)], label="label " + str(int(label)))
        else:
            plt.plot(range(len(labels)), occurences[int(label - 1)], label='ruis')
    plt.legend(loc="upper left")
    plt.xlabel("Tijdsreeksen")
    plt.ylabel("Aantal labels")
    plt.title("Evolutie van verschillende labels bij het toekomen van tijdsreeksen")
    plt.show()

def plot_label_distribution(series, labels, distance_matrix):
    sns.set_palette("bright")
    sns.set(style='whitegrid')
    # similarity_matrix = np.exp(- distance_matrix ** 2 / (2 ** 2))
    # plot_distribution_distance_of_labels(labels, distance_matrix)
    plot_evolution_of_distribution_of_labels(labels)
    # plot_distribution_distance_of_labels(labels, similarity_matrix, "gelijkheid", max=300)

def plot_DTW_of_TimeSeries_example():
    fig, ax = plt.subplots(nrows=1, ncols=3)
    s1 = stats.zscore(np.array([0,0,0,-1,2,3,4,3,5,-1]))
    s2 = stats.zscore(np.array([-1,2,3,4,3,5,-1]))
    s3 = stats.zscore(np.array([0,0,-1,2,3,4,3,5,-1]))
    # path = dtw.warping_path(s1, s2)
    ax[0].set_xlabel('t')
    ax[1].set_xlabel('t')
    ax[2].set_xlabel('t')
    ax[0].set_title('tijdsreeks 1')
    ax[1].set_title('tijdsreeks 2')
    ax[2].set_title('tijdsreeks 3')
    dtw12 = dtw.distance(s1, s2)
    dtw32 = dtw.distance(s3, s2)
    dtw13 = dtw.distance(s1, s3)
    x_title2 = str(dtw12) + " =< " + str(dtw13) + " + " + str(dtw32) + " = " + str(dtw13+dtw32)
    print(x_title2)
    ax[0].plot(range(len(s1)), s1, linewidth=3)
    ax[1].plot(range(len(s2)), s2, linewidth=3)
    ax[2].plot(range(len(s3)), s3, linewidth=3)
    #
    # dtwvis.plot_warping(s1, s2, path, fig=fig, axs=ax, filename="dtw_plot.png",
    #                     warping_line_options={'linewidth': 0.1, 'color': 'white', 'alpha': 0.8})


    # dtwvis.plot_warping(s1, s2, path, fig=fig, axs=ax[:,1], filename="dtw_plot.png",
    #                     warping_line_options={'linewidth': 0.1, 'color': 'white', 'alpha': 0.8})

    plt.show()
distance_matrix = np.loadtxt("distance_matrices/"+name+'_DM_nn.csv', delimiter=',')
series, labels, distance_matrix = make_new_datasets.modify_data(series, labels, distance_matrix, "stagnate")
plot_label_distribution(series, labels, distance_matrix)