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

def calculateAndSaveDM(series):
    dm = np.loadtxt('CBF_DM_nn.csv', delimiter=',')
    # dm = np.zeros((len(series), len(series)))
    # np.savetxt('CBF_DM_nn.csv', dm, delimiter=',')
    print(dm.shape)
    for i in range(0, len(series)):
        for j in range(0, len(series)):
            if j > i:
                dm[i][j] = dtw.distance(series[i], series[j])
            elif j == i:
                dm[i][j] = 0
            else:
                dm[i][j] = dm[j][i]
            print(i,j)
        np.savetxt('CBF_DM_nn.csv', dm, delimiter=',')

def plt_ari(ari_score_approx, ari_score_dtw, labels):

    boxes = [ari_score_approx[range(0, len(labels)), j] for j in range(len(labels))]
    print(len(labels), len(boxes))

    plt.title('Results for spectral clustering, starting from 890 time series with rank 20')
    plt.xlabel('Amount of time series')
    plt.ylabel('ARI-score')
    plt.plot(np.arange(len(labels))+1 , ari_score_dtw, color='r')
    plt.boxplot(boxes, labels=labels)
    plt.show()



def plot_time_complexity(dtw_calculations, labels):
    calc_full = []
    mean_approx_dtws = []
    for i in labels:
        calc_full.append(int(i*i/2)-i)

    for i in range(len(labels)):
        mean_approx_dtws.append(np.mean(dtw_calculations[:,i]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(labels, calc_full, color='r', label="Volledige afstandsmatrix")
    ax.plot(labels, mean_approx_dtws, color='b', label="Benadering van 890 tijdsreeksen")
    plt.title('Snelheid van algoritme (rang = 15)')
    plt.xlabel('Aantal tijdsreeksen')
    plt.ylabel('Aantal DTW-berekeningen')
    plt.legend(loc="upper left")
    plt.show()

def plt_errors(errors, labels):
    boxes = [errors[range(0, len(errors)), j] for j in range(len(errors[0, :]))]
    print(len(labels), len(boxes))

    plt.title('Results for spectral clustering, starting from 890 time series with rank 15')
    plt.xlabel('Amount of time series')
    plt.ylabel('Matrix error')
    plt.boxplot(boxes, labels=labels)
    plt.show()

# plt_box_plots("results/CBF/clustering/400/method1/spectral.csv", range(400, 900, 25))
# plt_box_plots("results/CBF/clustering/800/method2/spectral.csv", range(850, 900, 2))

def plt_skeletons(amount_of_skeletons, labels):
    mean_skeletons = []
    for i in range(len(labels)):
        mean_skeletons.append(np.mean(amount_of_skeletons[:, i]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(labels, mean_skeletons, color='b', label="Benadering van 890 tijdsreeksen")
    plt.title('Snelheid van algoritme (rang = 15)')
    plt.xlabel('Aantal tijdsreeksen')
    plt.ylabel('Aantal DTW-berekeningen')
    plt.legend(loc="upper left")
    plt.show()


def full_plot_of_method(filename, dtw_ari_filename):
    all_data =  np.load(filename)
    amount_of_ts = all_data[0,0,:]
    amount_of_skeletons = all_data[:,1,:]
    relative_error = all_data[:,2,:]
    all_ari_scores = all_data[:,3,:]
    all_dtw_calculations = all_data[:,4,:]

    ari_score_full_dtw_matrix = np.loadtxt(dtw_ari_filename, delimiter=',')

    plot_time_complexity(all_dtw_calculations, amount_of_ts)
    plt_ari(all_ari_scores, ari_score_full_dtw_matrix, amount_of_ts)
    plt_errors(relative_error, amount_of_ts)
    plt_skeletons(amount_of_skeletons, amount_of_ts)


dtw_ari_filename = "results/CBF/full/full_dtw_ari_400.csv"
filename = "results/CBF/full/method1/400_spectral_limited_rank.npy"
full_plot_of_method(filename, dtw_ari_filename)

dtw_ari_filename = "results/CBF/full/full_dtw_ari_400.csv"
filename = "results/CBF/full/method2/400_spectral_limited_rank.npy"
full_plot_of_method(filename, dtw_ari_filename)

dtw_ari_filename = "results/CBF/full/full_dtw_ari_400.csv"
filename = "results/CBF/full/method3/400_spectral_limited_rank.npy"
full_plot_of_method(filename, dtw_ari_filename)