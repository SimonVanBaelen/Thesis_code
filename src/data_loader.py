"""
###################################################################################################################
    All functions in this file were made by Mathias Pede, some functions that were unnecessary were removed.
    The original code can be found in [1] and was published alongside [2].

    [1]: M. Pede. Fast-time-series-clustering, 2020.
    https://github.com/MathiasPede/Fast-Time-Series-Clustering Accessed: (October 23,2022).

    [2]: M. Pede. Snel clusteren van tijdreeksen via lage-rang benaderingen. Master’s
    thesis, Faculteit Ingenieurswetenschappen, KU Leuven, Leuven, Belgium, 2020.
###################################################################################################################
"""

import numpy as np

def load_timeseries_from_tsv(path):
    """
    Loads Time Series from TSV file. The Format is expected to be the Class number as first element of the row,
    followed by the the elements of the time series.
    @param path:
    @return:
    """
    data = np.genfromtxt(path, delimiter='\t')
    labels, series = data[:, 0], data[:, 1:]
    return labels, series


def load_timeseries_from_multiple_tsvs(*args):
    all_labels = []
    all_series = []
    for path in args:
        labels, series = load_timeseries_from_tsv(path)
        all_labels.append(labels)
        all_series.append(series)
    result_labels = np.concatenate(all_labels)
    result_series = np.concatenate(all_series)
    return result_labels, result_series



