import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from src.cluster_problem import ClusterProblem
from src.data_loader import load_timeseries_from_tsv
from src.aca import ACA
from dtaidistance import dtw, clustering
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_box_plots(data, title, labels, method_names,full_dtw_ari=None):
    df = []
    methods = method_names
    indices = range(400, 900, 25)
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i], columns=list(indices)).assign(Methode=methods[i]))

    df = pd.concat(df)  # CONCATENATE
    df = pd.melt(df, id_vars=['Methode'], var_name=['Number'])
    sns.set_palette("deep")
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    # background_color = "aliceblue"
    ax = sns.boxplot(x="Number", y="value", hue="Methode", data=df,
                     flierprops=dict(marker='o', markerfacecolor='None', markersize=3, markeredgecolor='black'),
                     showfliers=False)  # RUN PLOT
    ax.set(xlabel=labels[0], ylabel=labels[1])
    ax.set_title(title)

    if not full_dtw_ari is None:
        sns.lineplot(full_dtw_ari, color="r", label="Volledige DTW afstandslatrix", linewidth=2)


def plot_two_line_plots(data1, data2, title, labels, method_names, ax=None):
    # fig, ax = plt.subplots(2,3)
    if ax is None:
        axis = plt
    else:
        axis = ax

    colors = ["steelblue", "sandybrown", "mediumseagreen", "indianred", "mediumpurple"]
    for i in range(len(data1)):
        mean_data = []
        indices = range(400, 900, 25)
        for j in range(len(data1[0][0,:])):
            mean_data.append(np.median(data2[i][:,j])/np.median(data1[i][:,j]))
        axis.plot(indices, mean_data, color=colors[i], label=method_names[i], linewidth=2)

    axis.legend(loc="lower left")
    if not ax is None:
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        axis.set_title(title)
    else:
        axis.xlabel(labels[0])
        axis.ylabel(labels[1])
        axis.title(title)


def full_plot_all_methods_seperate(filenames, method_names, dtw_file_name):
    amount_of_ts = np.load(filenames[0])[0,0,:]
    amount_of_skeletons = []
    relative_error = []
    all_ari_scores = []
    all_dtw_calculations = []

    for filename in filenames:
        all_data = np.load(filename)
        amount_of_skeletons.append(all_data[:,1,:])
        relative_error.append(all_data[:,2,:])
        all_ari_scores.append(all_data[:,3,:])
        all_dtw_calculations.append(all_data[:,4,:])

    full_dtw_ari = np.loadtxt(dtw_file_name, delimiter=",")
    title1 = "Kwaliteit clustering bij gebruik verschillende methodes met homogene labels"
    title2 = "Gemiddelde relatieve benaderingsfout bij gebruik verschillende methodes met homogene labels"
    title3 = "Tijdscomplexiteit van verschillende methodes"
    title4 = "Ruimtecomplexiteit van verschillende methodes"

    labels1 = ["Aantal tijdsreeksen", "ARI-score"]
    labels2 = ["Aantal tijdsreeksen", "Relatieve benaderingsfout"]
    labels3 = ["Aantal tijdsreeksen", "Relatieve fout gedeeld door aantal DTW-berekeningen"]
    labels4 = ["Aantal tijdsreeksen", "Relatieve fout gedeeld door aantal skeletten"]
    plot_box_plots(all_ari_scores, title1, labels1, method_names, full_dtw_ari)
    plt.show()
    plt.cla()

    plot_box_plots(relative_error, title2, labels2, method_names)
    plt.show()
    plt.cla()

    fig, ax = plt.subplots(1,2)
    plot_two_line_plots(all_dtw_calculations, relative_error, title3, labels3, method_names, ax[0])
    plot_two_line_plots(amount_of_skeletons, relative_error, title4, labels4, method_names, ax[1])
    plt.show()
    plt.cla()

def full_plot_all(filenames):
    ae_ari = np.array([])
    ae_error = np.array([])
    for filename in filenames:
        all_data = np.load(filename)
        ae_ari = np.append(ae_ari, all_data[:, 3, :])
        ae_error = np.append(ae_error, all_data[:, 2, :])
    plt_error_ari(ae_error, ae_ari)

dtw_ari_filename1 = "results/CBF/full_dtw_ari_400.csv"
filename1 = "results/CBF/combined_full/method1_full.npy"
filename2 = "results/CBF/combined_full/method2_full.npy"
filename3 = "results/CBF/combined_full/method3_full.npy"
filename4 = "results/CBF/combined_full/method4_full.npy"
filename5 = "results/CBF/combined_full/method5_full.npy"

names = [filename1, filename2, filename3, filename4, filename5]
method_names = ["extend skeletons", "adaptive add skeletons", "reevaluate pivots", "extend matrix", "non-adaptive add skeleton"]
full_plot_all_methods_seperate([names[1], names[2]], [method_names[1], method_names[2]], dtw_ari_filename1)
full_plot_all_methods_seperate([names[3], names[4]], [method_names[3], method_names[4]], dtw_ari_filename1)
full_plot_all_methods_seperate(names, method_names, dtw_ari_filename1)