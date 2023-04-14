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

# FacesUCR, FaceAll
names = ["CBF"]

name = names[0]
path_train = "Data/" + name + "/" + name + "_TRAIN.tsv"
labels_train, series_train = load_timeseries_from_tsv(path_train)

path_test = "Data/" + name + "/" + name + "_TEST.tsv"
labels_tr, series_tr = load_timeseries_from_tsv(path_test)
print(len(series_tr[10]), len(series_tr))
print(len(series_train[10]), len(series_train))
# series_tr = np.concatenate((series_tr, series_train))
# labels_tr = np.concatenate((labels_tr, labels_train))

func_name = "dtw"         # "msm"/"ed" other options
args = {"window": len(series_tr[0])-1}   # for MSM "c" parameter

# rank = 40


def find_error(actual_dm, approx_dm):
    diff = np.subtract(actual_dm, approx_dm)
    return norm(diff)

def normalize(series_tr):
    # for serie in series_tr:
    #     mi = min(serie)
    #     ma = max(serie)
    #     for i in range(0,len(serie)):
    #         serie[i] = (serie[i]-mi)/(ma-mi)
    return stats.zscore(series_tr)

def calculateClusters(true, approx, index, labels, k):
    temp_true = np.exp(- true ** 2 / (2. * 3 ** 2))
    temp_approx = np.exp(- approx ** 2 / (2. * 3 ** 2))

    # temp_true = true
    # temp_approx = approx


    # model_spec = AgglomerativeClustering(n_clusters=k, linkage='ward', affinity='euclidean')
    model_spec = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0)
    result_spec = model_spec.fit_predict(temp_true)
    normDTW = adjusted_rand_score(labels[0:index], result_spec)
    cm1 = confusion_matrix(labels[0:index], result_spec)


    # model_spec = AgglomerativeClustering(n_clusters=k, linkage='ward', affinity='euclidean')
    model_spec = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='discretize', random_state=0)
    result_spec = model_spec.fit_predict(temp_approx)
    normApprox = adjusted_rand_score(labels[0:index], result_spec)
    cm2 = confusion_matrix(labels[0:index], result_spec)
    # print(cm1, cm2)
    return normDTW, normApprox


def saveNorm(normDTW, normApprox, results, index):
    results[0,index] = normDTW
    results[len(results)-1, index] = normApprox
    return index+1

def add_series_to_dm(true, next, dm):
    next = next - 1
    all_dtw = np.transpose(dm[next, range(next + 1)])
    true = np.append(true, [np.zeros(len(true[1, :]))], 0)
    true = np.append(true, np.transpose([np.zeros(len(true[1, :]) + 1)]), 1)

    true[:, next] = all_dtw
    true[next, :] = all_dtw
    return true


def rearrange_data(labels, series, full_dm):
    indices1 = np.where(labels == 1)[0][range(0,250)]
    indices2 = np.where(labels == 2)[0][range(0,25)]
    indices12 = np.concatenate((indices1,indices2))
    np.random.shuffle(indices12)
    indices3 = np.where(labels == 1)[0][range(250,265)]
    all_indices = np.concatenate((indices12,indices3))

    new_series = series[all_indices]
    new_labels = labels[all_indices]
    new_dm = full_dm[all_indices, :]
    new_dm = new_dm[:, all_indices]

    return new_labels, new_series, new_dm


def cluster_stream(labels, series, fileName):
    normalize(series)
    full_dm = np.loadtxt('CBF_DM_nn.csv', delimiter=',')
    # labels, series, full_dm = rearrange_data(labels, series, full_dm)
    k = len(set(labels))
    start_index = 400
    skip = 25
    while True:

        true = full_dm[range(start_index), :]
        true = true[:, range(start_index)]
        try:
            results_ari = np.loadtxt(fileName, delimiter=',')
            results_ari = np.append(results_ari, [np.zeros(len(results_ari[0, :]))], 0)
        except:
            results_ari = np.zeros((2, int((len(series) - start_index)/skip)))
            # print("yes")
            # results_ari = np.zeros((1, int((len(series) - start_index) / skip)))

        cp = ClusterProblem(series[0:start_index], func_name, compare_args=args)
        approx = ACA(cp, tolerance=0.05, max_rank=15)

        index = start_index
        save_index = 0
        normDTW, normApprox = calculateClusters(true, approx.approx, index, labels, k)
        save_index = saveNorm(normDTW, normApprox, results_ari, save_index)
        print("Iteration " + str(index), "DTW ARI:", normDTW, "Confusion matrix", 0)
        print("Iteration " + str(index), "Approx ARI:", normApprox, "Confusion matrix")
        # error = np.linalg.norm(true - approx.approx)
        # results_ari[len(results_ari) - 1, save_index] = error
        # print(error)
        # save_index += 1
        while index < len(series)-1:
            index += 1
            true = add_series_to_dm(true, index, full_dm)
            approx.extend(series[index], full_dm, method="v1")
            if index % skip == 0:
                normDTW, normApprox = calculateClusters(true, approx.approx, index, labels, k)
                save_index = saveNorm(normDTW, normApprox, results_ari, save_index)
                print("Iteration " + str(index), "DTW ARI:", normDTW, "Confusion matrix", 0)
                print("Iteration " + str(index), "Approx ARI:", normApprox, "Confusion matrix")
                # error = np.linalg.norm(true - approx.approx)
                # results_ari[len(results_ari) - 1, save_index] = error
                # print(error)
                # save_index += 1

        np.savetxt(fileName,results_ari, delimiter=',')
        if len(results_ari) > 100:
            break

def calculateAndSaveDM(series):
    dm = np.loadtxt('CBF_DM_nn.csv', delimiter=',')
    # dm = np.zeros((len(series), len(series)))
    # np.savetxt('CBF_DM_nn.csv', dm, delimiter=',')
    print(dm.shape)
    for i in range(4, len(series)):
        for j in range(0, len(series)):
            if j > i:
                dm[i][j] = dtw.distance(series[i], series[j])
            elif j == i:
                dm[i][j] = 0
            else:
                dm[i][j] = dm[j][i]
            print(i,j)
        np.savetxt('CBF_DM_nn.csv', dm, delimiter=',')

def plt_box_plots(fileName):
    # labels = range(890, 899, 1)
    labels = range(400, 900, 25)
    print(len(labels))
    results = np.loadtxt(fileName, delimiter=',')

    boxes = [results[range(1, len(results)), j] for j in range(len(results[0, :])-1)]
    print(len(labels), len(boxes))

    plt.title('Results for spectral clustering, starting from 890 time series with rank 15')
    plt.xlabel('Amount of time series')
    plt.ylabel('ARI-score')
    plt.plot(np.arange(len(results[0,:])-1)+1 , results[0,range(len(results[0])-1)], color='r')
    plt.boxplot(boxes, labels=labels)
    plt.show()


# series_tr = normalize(series_tr)
# calculateAndSaveDM(series_tr)
# plt_results_cbf_450()
# plt_results_cbf_450()
fileName = 'results_ari_400_spec_precomputed.csv'
# fileName2 = 'ERROR2.csv'
cluster_stream(labels_tr, series_tr, fileName)
plt_box_plots(fileName)

def plot_time_complexity():
    x = range(0, 900, 1)
    y_full = []
    # y_400 = []
    # y_750 = []
    y_ex = []
    for i in x:
        y_full.append(int(i*i/2)-i)
        y_ex.append(i*15)
        # if i < 400:
        #     y_400.append(i*15)
        # else:
        #     y_400.append(i * 15 + int(((i-400)**2)/2))
        #
        # if i < 750:
        #     y_750.append(i*15)
        # else:
        #     y_750.append(i * 15 + int(((i-750)**2)/2))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_full, color='r', label="Volledige afstandsmatrix")
    ax.plot(x, y_ex, color='b', label="Benadering van 890 tijdsreeksen")
    # ax.plot(x, y_400, color='b', label="Benadering van 400 tijdsreeksen")
    # ax.plot(x, y_750, color='g', label="Benadering van 750 tijdsreeksen")
    plt.title('Snelheid van algoritme (rang = 15)')
    plt.xlabel('Aantal tijdsreeksen')
    plt.ylabel('Aantal DTW-berekeningen')
    plt.legend(loc="upper left")
    plt.show()


def plt_errors(fileName):
    labels = range(890, 899, 1)
    # labels = range(400, 900, 25)
    print(len(labels))
    results = np.loadtxt(fileName, delimiter=',')
    for i in range(len(labels)):
        results[:,i] = results[:,i]
    boxes = [results[range(0, len(results)), j] for j in range(len(results[0, :])-1)]
    print(len(labels), len(boxes))

    plt.title('Results for spectral clustering, starting from 890 time series with rank 15')
    plt.xlabel('Amount of time series')
    plt.ylabel('Matrix error')
    plt.boxplot(boxes, labels=labels)
    plt.show()

plot_time_complexity()

# plt_box_plots(fileName)
# plt_errors(fileName2)