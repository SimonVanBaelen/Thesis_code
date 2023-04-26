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

def plt_ari(ari_score_approx, ari_score_dtw, labels, method):
    boxes = [ari_score_approx[range(0, len(ari_score_approx)), j] for j in range(len(labels))]

    plt.title('Resultaten spectraal clusteren voor ' + method)
    plt.xlabel('Aantal tijdsreeksen')
    plt.ylabel('ARI-score')
    plt.plot(np.arange(len(labels))+1 , ari_score_dtw, color='r')
    plt.boxplot(boxes, labels=labels)
    plt.show()



def plot_time_complexity(dtw_calculations, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['c', 'm', 'b', 'g']
    lab = ["Methode 1", "Methode 2", "Methode 3", "Methode 4"]
    calc_full = []
    for i in labels:
        calc_full.append(int(i*i/2)-i)
    ax.plot(labels, calc_full, color='r', label="Volledige afstandsmatrix")
    for method in range(len(dtw_calculations)):
        mean_approx_dtws = []
        for i in range(len(labels)):
            mean_approx_dtws.append(np.mean(dtw_calculations[method][:,i]))
        ax.plot(labels, mean_approx_dtws, color=colors[method], label=lab[method])
    if labels[0] < 500:
        plt.title('Snelheid van algoritme (rang = 20)')
    else:
        plt.title('Snelheid van algoritme (rang = 40)')
    plt.xlabel('Aantal tijdsreeksen')
    plt.ylabel('Aantal DTW-berekeningen')
    plt.legend(loc="upper left")
    plt.show()

def plt_errors(errors, labels, method):
    boxes = [errors[range(0, len(errors)), j] for j in range(len(labels))]
    plt.title('Benaderingsfout met ' + method)
    plt.xlabel('Aantal tijdsreeksen')
    plt.ylabel('Relatieve benaderingsfout')
    plt.boxplot(boxes, labels=labels)
    plt.show()


# def plt_skeletons(amount_of_skeletons, labels):
#     mean_skeletons = []
#     for i in range(len(labels)):
#         mean_skeletons.append(np.mean(amount_of_skeletons[:, i]))
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     ax.plot(labels, mean_skeletons, color='b', label="Benadering van 890 tijdsreeksen")
#     plt.title('Snelheid van algoritme (rang = 15)')
#     plt.xlabel('Aantal tijdsreeksen')
#     plt.ylabel('Aantal DTW-berekeningen')
#     plt.legend(loc="upper left")
#     plt.show()


def plt_error_ari(relative_error, all_ari_scores):
    error = relative_error.flatten()
    ari_scores = all_ari_scores.flatten()
    # ari_scores = [x for _, x in sorted(zip(error, ari_scores))]
    # error.sort()
    error_for_plot = np.arange(0.20, 1, 0.01)
    ari_for_plot = []
    for m in error_for_plot:
        all_inbetween = []
        for i in range(len(error)):
            if m - 0.1 < error[i] < m:
                all_inbetween.append(ari_scores[i])
        ari_for_plot.append(np.mean(all_inbetween))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(error_for_plot, ari_for_plot, color='b')
    plt.title('Verband benaderingsfout en ari-score')
    plt.xlabel('Relatieve benaderingsfout')
    plt.ylabel('ari-score')
    plt.legend(loc="upper left")
    plt.show()

def plot_timeseries(series_tr, labels_tr):
    for i in range(len(series_tr)):
        x = [x for x in range(0, len(series_tr[i]))]
        print(labels_tr[i])
        colors = ["g", "r", "b"]
        plt.plot(x, series_tr[i], linewidth=1.5, color=colors[int(labels_tr[i]) - 1])
        plt.xlabel('t')
        name = "pictures/ts_" + str(i) + "_" + str(labels_tr[i]) + ".png"
        plt.savefig(name, transparent=True)
        plt.show()


def full_plot_of_method(filename, method, dtw_ari_filename):
    all_data =  np.load(filename)
    amount_of_ts = all_data[0,0,:]
    amount_of_skeletons = all_data[:,1,:]
    relative_error = all_data[:,2,:]
    all_ari_scores = all_data[:,3,:]
    all_dtw_calculations = all_data[:,4,:]

    ari_score_full_dtw_matrix = np.loadtxt(dtw_ari_filename, delimiter=',')

    # plt_ari(all_ari_scores, ari_score_full_dtw_matrix, amount_of_ts, method)
    # plt_errors(relative_error, amount_of_ts, method)


def plot_mean_all_ari(all_ari_scores, dtw_file_name, labels):
    ari_score_full_dtw_matrix = np.loadtxt(dtw_file_name, delimiter=',')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['c', 'm', 'b', 'g']
    lab = ["Methode 1", "Methode 2", "Methode 3", "Methode 4"]
    plt.plot(labels , ari_score_full_dtw_matrix, color='r')
    for method in range(len(all_ari_scores)):
        mean_approx_dtws = []
        for i in range(len(labels)):
            mean_approx_dtws.append(np.mean(all_ari_scores[method][:, i]))
        ax.plot(labels, mean_approx_dtws, color=colors[method], label=lab[method])
    if labels[0] < 500:
        plt.title('Snelheid van algoritme (rang = 20)')
    else:
        plt.title('Snelheid van algoritme (rang = 40)')
    plt.xlabel('Aantal tijdsreeksen')
    plt.ylabel('Aantal DTW-berekeningen')
    plt.legend(loc="upper left")
    plt.show()


def full_plot_all_methods(filenames, dtw_file_name):
    amount_of_ts = np.load(filenames[0])[0,0,:]
    amount_of_skeletons = []
    relative_error = []
    all_ari_scores = []
    all_dtw_calculations = []

    for filename in filenames:
        all_data = np.load(filename)
        amount_of_skeletons.append(all_data[:,1,:])
        relative_error = np.append(relative_error, all_data[:,2,:])
        all_ari_scores.append(all_data[:,3,:])
        all_dtw_calculations.append(all_data[:,4,:])

    plot_time_complexity(all_dtw_calculations, amount_of_ts)
    plot_mean_all_ari(all_ari_scores, dtw_file_name, amount_of_ts)


def full_plot_all(filenames):
    ae_ari = np.array([])
    ae_error = np.array([])
    for filename in filenames:
        all_data = np.load(filename)
        ae_ari = np.append(ae_ari, all_data[:, 3, :])
        ae_error = np.append(ae_error, all_data[:, 2, :])
    plt_error_ari(ae_error, ae_ari)

dtw_ari_filename1 = "results/CBF/full/full_dtw_ari_400.csv"
dtw_ari_filename2 = "results/CBF/full/full_dtw_ari_800.csv"
filename1 = "results/CBF/full/method1/400_spectral_limited_rank.npy"
filename2 = "results/CBF/full/method2/400_spectral_limited_rank.npy"
filename3 = "results/CBF/full/method3/400_spectral_limited_rank.npy"
filename4 = "results/CBF/full/method4/400_spectral_limited_rank.npy"
# filename5 = "results/CBF/full/method1/800_spectral_limited_rank.npy"
# filename6 = "results/CBF/full/method2/800_spectral_limited_rank.npy"
# filename7 = "results/CBF/full/method3/800_spectral_limited_rank.npy"
# filename8 = "results/CBF/full/method4/800_spectral_limited_rank.npy"
# full_plot_of_method(filename1, "methode 1", dtw_ari_filename1)
# full_plot_of_method(filename2, "methode 2",dtw_ari_filename1)
# full_plot_of_method(filename3, "methode 3",dtw_ari_filename1)
# full_plot_of_method(filename4, "methode 4",dtw_ari_filename1)
# full_plot_of_method(filename5, "methode 1",dtw_ari_filename2)
# full_plot_of_method(filename6, "methode 2",dtw_ari_filename2)
# full_plot_of_method(filename7, "methode 3",dtw_ari_filename2)
# full_plot_of_method(filename8, "methode 4",dtw_ari_filename2)
filename5 = "results/CBF/full/method1/400_spectral_unlimited_rank.npy"
filename6 = "results/CBF/full/method2/400_spectral_unlimited_rank.npy"
filename7 = "results/CBF/full/method3/400_spectral_unlimited_rank.npy"
filename8 = "results/CBF/full/method4/400_spectral_unlimited_rank.npy"

names1 = [filename1, filename2, filename3, filename4]
names2 = [filename5, filename6, filename7, filename8]

full_plot_all_methods(names1, dtw_ari_filename1)
full_plot_all_methods(names2, dtw_ari_filename1)
# full_plot_all_methods(names2)
# full_plot_all(names1+names2)