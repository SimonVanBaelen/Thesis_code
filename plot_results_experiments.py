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


def plot_box_plots(data, title, labels, method_names, indices, full_dtw_ari=None, ax=None):
    df = []
    methods = method_names
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i], columns=list(indices)).assign(Methode=methods[i]))

    df = pd.concat(df)  # CONCATENATE
    df = pd.melt(df, id_vars=['Methode'], var_name=['Number'])
    print(df)
    sns.set_palette("deep")
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    # background_color = "aliceblue"
    ax = sns.boxplot(x="Number", y="value", hue="Methode", data=df,
                     flierprops=dict(marker='o', markerfacecolor='None', markersize=3, markeredgecolor='black'),
                     showfliers=False, ax=ax)  # RUN PLOT
    ax.set(xlabel=labels[0], ylabel=labels[1])
    ax.set_title(title)

    if not full_dtw_ari is None:
        sns.lineplot(full_dtw_ari, color="r", label="Volledige DTW afstandslatrix", linewidth=2)


def plot_two_line_plots(data1, data2, title, labels, method_names, indices,  ax=None):
    # fig, ax = plt.subplots(2,3)
    if ax is None:
        axis = plt
    else:
        axis = ax

    colors = ["steelblue", "sandybrown", "mediumseagreen", "indianred", "mediumpurple"]
    for i in range(len(data1)):
        mean_data = []
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


def plot_multiple_plots_seperate(data1, data2, title, subtitles, labels, method_names, indices, y_lim1, y_lim2):
    fig, ax = plt.subplots(2,3)
    colors2 = ["steelblue", "sandybrown", "mediumseagreen", "indianred", "mediumpurple"]
    for i in range(len(data1)):
        mean_data1 = []
        mean_data2 = []
        for j in range(len(data1[0][0, :])):
            mean_data1.append(np.median(data1[i][:, j]))
            mean_data2.append(np.median(data2[i][:, j]))
        x = 1 if i > 2 else 0
        ax[x, int(i%3)].plot(indices, mean_data2, color="red", label=method_names[i], linewidth=2)
        ax[x, int(i%3)].set_xlabel(labels[0])
        ax[x, int(i%3)].set_ylabel(labels[1])
        if i < len(subtitles):
            ax[x, int(i % 3)].set_title(subtitles[i])
        ax[x, int(i % 3)].set_ylim(y_lim2)
        ax2 = ax[x, int(i % 3)].twinx()
        ax2.set_ylim(y_lim1)
        ax2.set_ylabel("Aantal skeletten")
        ax2.plot(indices, mean_data1, color=colors2[i], label=method_names[i], linewidth=2)
    for i in range(len(data1)):
        mean_data1 = []
        mean_data2 = []
        for j in range(len(data1[0][0, :])):
            mean_data1.append(np.median(data1[i][:, j]))
            mean_data2.append(np.median(data2[i][:, j]))
        ax[1,2].plot(indices, mean_data1, color=colors2[i], label=method_names[i], linewidth=2)
    # fig.set_title(title)

    # axis.legend(loc="lower left")
    # if not ax is None:
    #     axis.set_xlabel(labels[0])
    #     axis.set_ylabel(labels[1])
    #     axis.set_title(title)
    # else:
    #     axis.xlabel(labels[0])
    #     axis.ylabel(labels[1])
    #     axis.title(title)


def plot_median_error(ax, relative_error, colors,  method_names, indices, y_lim=[0.05, 0.46], linestyle = None,
                      legend_title="Errors", full_dtw_calc=None):
    t = []
    if not y_lim is None:
        ax.set_ylim(y_lim)
    for i in range(len(relative_error)):
        median_data = []
        for j in range(len(relative_error[0][0, :])):
            median_data.append(np.median(relative_error[i][:, j]))
        if not linestyle is None:
            t.append(ax.plot(indices, median_data, color=colors[i], label=method_names[i], linewidth=2, linestyle=linestyle))
            ax.legend(title="Relatieve benaderingsfout")
            ax.set_ylabel("Relatieve benaderingsfout")
        else:
            t.append(ax.plot(indices, median_data, color=colors[i], label=method_names[i], linewidth=2))
            ax.legend(title=legend_title)
            ax.set_ylabel("Aantal " + legend_title)
        if not full_dtw_calc is None:
            ax.plot(indices, full_dtw_calc, color="r", label="Volledige afstandsmatrix", linewidth=2)
            full_dtw_calc = None
        ax.set_xlabel("Aantal tijdsreeksen")

def plot_multiple_plots_together(all_dtw_calculations, relative_error, title, method_names, labels, indices, y_lim, full_dtw_calc=None):
    fig, ax = plt.subplots(1, 3)
    colors2 = ["steelblue", "sandybrown", "mediumseagreen", "indianred", "mediumpurple"]
    ax2 = [ax[0].twinx(), ax[1].twinx()]
    plot_box_plots(all_dtw_calculations[0:3], "test", labels[0:3], method_names[0:3], indices, ax=ax[0])
    plot_box_plots(all_dtw_calculations[3:len(relative_error)], "test", labels, method_names[3:len(relative_error)], indices, ax=ax[1])
    plot_median_error(ax2[0], relative_error[0:3], colors2[0:3],  method_names[0:3], indices)
    plot_median_error(ax2[1], relative_error[3:len(relative_error)], colors2[3:len(relative_error)], method_names[3:len(relative_error)], indices)
    plot_median_error(ax[2], all_dtw_calculations, colors2, method_names, indices,y_lim=y_lim)
    if not full_dtw_calc is None:
        sns.lineplot(full_dtw_calc, color="r", label="Volledige afstandsmatrix", linewidth=2, ax=ax[2])
    # plt.legend(title='Smoker', loc='upper left')
    plt.show()
    # plt.cla()


def plot_skeletons(amount_of_skeletons, relative_error, title, method_names, labels4, indices, y_lim, full_dtw_calc=None):
    fig, ax = plt.subplots(1, 3)
    fig.suptitle(title)
    colors2 = ["steelblue", "sandybrown", "mediumseagreen", "indianred", "mediumpurple"]
    ax2 = [ax[0].twinx(), ax[1].twinx()]
    if y_lim[1] < 1000:
        legend_title = "Skeletten"
        plot_median_error(ax[0], amount_of_skeletons[0:3], colors2[0:3], method_names[0:3], indices, y_lim=[20, 140], legend_title=legend_title)
    else:
        legend_title = "DTW-calculaties"
        plot_median_error(ax[0], amount_of_skeletons[0:3], colors2[0:3], method_names[0:3], indices, y_lim=[0, 450000],legend_title=legend_title)
    plot_median_error(ax[1], amount_of_skeletons[3:len(relative_error)], colors2[3:len(relative_error)],method_names[3:len(relative_error)], indices, y_lim=None, legend_title=legend_title)
    plot_median_error(ax2[0], relative_error[0:3], colors2[0:3], method_names[0:3], indices, linestyle="--")
    plot_median_error(ax2[1], relative_error[3:len(relative_error)], colors2[3:len(relative_error)], method_names[3:len(relative_error)], indices, linestyle="--")
    plot_median_error(ax[2], amount_of_skeletons, colors2, method_names, indices, y_lim=y_lim, legend_title=legend_title, full_dtw_calc=full_dtw_calc)

    plt.show()


def full_plot_all_methods_seperate(filenames, method_names, dm_file_name=None):
    indices = [int(x) for x in np.load(filenames[0])[0,0,:]]
    print(indices)
    del indices[-1]
    amount_of_skeletons = []
    relative_error = []
    all_ari_scores = []
    all_dtw_calculations = []
    for filename in filenames:
        all_data = np.load(filename)
        all_data = all_data[range(all_data.shape[0]-1), :, :]
        all_data = all_data[:, :, range(all_data.shape[2]-1)]
        amount_of_skeletons.append(all_data[:,1,:])
        relative_error.append(all_data[:,2,:])
        all_ari_scores.append(all_data[:,3,:])
        all_dtw_calculations.append(all_data[:,4,:])


    title1 = "Kwaliteit clustering bij gebruik verschillende methodes"
    title2 = "Gemiddelde relatieve benaderingsfout bij gebruik verschillende methodes"
    title3 = "Tijdscomplexiteit van verschillende methodes samen met de relatieve benaderingsfout"
    title4 = "Ruimtecomplexiteit van verschillende methodes samen met de relatieve benaderingsfout"

    labels1 = ["Aantal tijdsreeksen", "ARI-score"]
    labels2 = ["Aantal tijdsreeksen", "Relatieve benaderingsfout"]
    labels3 = ["Aantal tijdsreeksen", "Relatieve fout"]
    labels4 = ["Aantal tijdsreeksen", "Relatieve fout"]

    if not dm_file_name is None:
        full_dtw_ari = np.load(dm_file_name)
        print(full_dtw_ari)
        plot_box_plots(all_ari_scores, title1, labels1, method_names, indices, full_dtw_ari=full_dtw_ari[0,1,:])
    else:
        plot_box_plots(all_ari_scores, title1, labels1, method_names, indices, full_dtw_ari=None)
    plt.show()
    plt.cla()

    plot_box_plots(relative_error, title2, labels2, method_names, indices)
    plt.show()
    plt.cla()

    # plot_multiple_plots_seperate(all_dtw_calculations, relative_error, title3, method_names, labels3, method_names, indices, y_lim1=[0, 600000], y_lim2=[0.1, 0.35])
    # plot_multiple_plots_seperate(amount_of_skeletons, relative_error, title4, method_names, labels4, method_names, indices, y_lim1=[40,615], y_lim2=[0.1,0.35])
    # plot_multiple_plots_together(all_dtw_calculations, relative_error, title3, method_names, labels3, indices, y_lim=[0,600000], full_dtw_calc=full_dtw_ari[0,2,:])
    plot_skeletons(all_dtw_calculations, relative_error, title3, method_names, labels3, indices, y_lim=[0, 700000], full_dtw_calc=full_dtw_ari[0,2,:])
    plot_skeletons(amount_of_skeletons, relative_error, title4, method_names, labels4, indices, y_lim=[0,600])
    # plot_multiple_plots_together(amount_of_skeletons, relative_error, title4, method_names, labels4, method_names, indices, y_lim1=[40,615], y_lim2=[0.1,0.35])

def full_plot_all(filenames):
    ae_ari = np.array([])
    ae_error = np.array([])
    for filename in filenames:
        all_data = np.load(filename)
        ae_ari = np.append(ae_ari, all_data[:, 3, :])
        ae_error = np.append(ae_error, all_data[:, 2, :])
    plt_error_ari(ae_error, ae_ari)

filename0 = "full_dm_ari/CBF_full_dm_results_53_400_sorted.npy"
filename1 = "results/part1/sorted/method1_full.npy"
filename2 = "results/part1/sorted/method2_full.npy"
filename3 = "results/part1/sorted/method3_full.npy"
filename4 = "results/part1/sorted/method4_full.npy"
filename5 = "results/part1/sorted/method5_full.npy"
# filename1 = "results/part1/homogeen_batches/25/_method1_25.npy"
# filename2 = "results/part1/homogeen_batches/25/_method2_25.npy"
# filename3 = "results/part1/homogeen_batches/25/_method3_25.npy"
# filename4 = "results/part1/homogeen_batches/25/_method4_25.npy"
# filename5 = "results/part1/homogeen_batches/25/_method5_25.npy"

names = [filename1, filename2, filename3, filename4, filename5]

method_names = ["Skelet update", "T.-g. additieve update", "Adaptieve update", "Exacte additieve update", "Maximale additieve update"]
# full_plot_all_methods_seperate([names[1], names[2]], [method_names[1], method_names[2]], dtw_ari_filename1)
# full_plot_all_methods_seperate([names[3], names[4]], [method_names[3], method_names[4]], dtw_ari_filename1)
full_plot_all_methods_seperate(names, method_names, dm_file_name=filename0)