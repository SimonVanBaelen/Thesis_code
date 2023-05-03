import pandas as pd

from src.data_loader import load_timeseries_from_tsv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

names = ["CBF"]

name = names[0]
path_test = "Data/" + name + "/" + name + "_TEST.tsv"
labels, series = load_timeseries_from_tsv(path_test)
distance_matrix = np.loadtxt('CBF_DM_nn.csv', delimiter=',')


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
            axes[i, j].set_xlim([0, 12])
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
        plt.plot(range(len(labels)), occurences[int(label-1)], label="label " + str(int(label)))

    plt.legend(loc="upper left")
    plt.xlabel("Tijdsreeksen")
    plt.ylabel("Aantal labels")
    plt.title("Evolutie van verschillende labels bij het toekomen van tijdsreeksen")
    plt.show()

def plot_label_distribution(series, labels, distance_matrix):
    sns.set_palette("bright")
    sns.set(style='whitegrid')
    # similarity_matrix = np.exp(- distance_matrix ** 2 / (2 ** 2))
    plot_distribution_distance_of_labels(labels, distance_matrix)
    plot_evolution_of_distribution_of_labels(labels)
    # plot_distribution_distance_of_labels(labels, similarity_matrix, "gelijkheid", max=300)


plot_label_distribution(series, labels, distance_matrix)