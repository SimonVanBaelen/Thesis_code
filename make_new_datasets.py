import numpy as np

def make_new_label_dataset(series, labels, full_dm):
    indices1 = np.where(labels == 1)[0][range(0, 200)]
    indices2 = np.where(labels == 2)[0][range(0, 200)]
    indices3 = np.where(labels == 3)[0][range(0, 200)]
    indices12 = np.concatenate((indices1, indices2))
    all_indices = np.concatenate((indices12, indices3))

    new_series = series[all_indices]
    new_labels = labels[all_indices]
    new_dm = full_dm[all_indices, :]
    new_dm = new_dm[:, all_indices]

    return new_series, new_labels, new_dm

def make_stagnate_dataset(series, labels, full_dm):
    start = 400
    start_indices1 = len(np.where(labels[0:start] == 1)[0])
    start_indices2 = len(np.where(labels[0:start] == 2)[0])
    print(start_indices1, start_indices2)
    indices1 = np.where(labels == 1)[0][range(start_indices1, len(np.where(labels == 1)[0]))]
    indices2 = np.where(labels == 2)[0][range(start_indices2, len(np.where(labels == 2)[0]))]
    all_indices = np.array(range(start))
    for i in range(min([len(np.where(labels == 1)[0])-start_indices1, len(np.where(labels == 2)[0])-start_indices2])):
        all_indices = np.concatenate((all_indices,[indices1[i]]))
        all_indices = np.concatenate((all_indices, [indices2[i]]))

    new_series = series[all_indices]
    new_labels = labels[all_indices]
    new_dm = full_dm[all_indices, :]
    new_dm = new_dm[:, all_indices]

    return new_series, new_labels, new_dm

def make_noise_dataset(series, labels, full_dm):
    start = 400
    amount_of_noise = 200
    all_indices = np.array(range(start + amount_of_noise))

    new_series = series[all_indices]
    new_labels = labels[all_indices]
    new_dm = full_dm[all_indices, :]
    new_dm = new_dm[:, all_indices]
    for i in range(start, start+amount_of_noise):
        new_labels[i] = "4"
        new_dm[i,:] = np.ones(len(new_dm)) + 8
        new_dm[:,i] = np.ones(len(new_dm)) + 8
    return new_series, new_labels, new_dm

def make_sorted_dataset(series, labels, full_dm):
    temp = []
    for i in range(len(full_dm)):
        temp.append([i, labels[i], np.mean(full_dm[i,:])])
    temp.sort(key=lambda x: x[2])
    temp = np.array(temp).astype(int)
    all_indices = temp[:,0]
    new_series = series[all_indices]
    new_labels = labels[all_indices]
    new_dm = full_dm[all_indices, :]
    new_dm = new_dm[:, all_indices]

    return new_series, new_labels, new_dm

def modify_data(series, labels, true_dm, modify_name=None):
    if modify_name == 'stagnate':
        series, labels, true_dm = make_stagnate_dataset(series, labels, true_dm)
    elif modify_name == 'noise':
        series, labels, true_dm = make_noise_dataset(series, labels, true_dm)
    elif modify_name == 'new':
        series, labels, true_dm = make_new_label_dataset(series, labels, true_dm)
    elif modify_name == 'sorted':
        series, labels, true_dm = make_sorted_dataset(series, labels, true_dm)
    return series, labels, true_dm
