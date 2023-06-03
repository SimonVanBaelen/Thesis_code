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
from dtaidistance.dtw import distance_fast, distance_matrix_fast
import numpy as np

"""
File containing the the distance/similarity functions to compare different time series
# Implemented functions:
#   Dynamic Time Warping (DTW)

"""
def dtw(s1, s2, args: dict):
    if 'window' in args:
        window = args['window']
    else:
        window=None
    return distance_fast(s1, s2, window=window)


def dtw_matrix(series, args: dict):
    if 'window' in args:
        window = args['window']
    else:
        window=None
    if 'block' in args:
        block = args['block']
    else:
        block= None
    if 'compact' in args:
        compact = args['compact']
    else:
        compact = False
    return distance_matrix_fast(series, window=window, block=block, compact=compact)


functions = {
    'dtw': (dtw, dtw_matrix)
}


def compute_distance(s1, s2, func, args=None):
    if args is None:
        args={}
    compare = functions[func][0]
    return compare(s1, s2, args)


def compute_row(series, row_index, func, args=None):
    if args is None:
        args={}
    compare = functions[func][1]

    size = len(series)
    args['compact'] = True

    # row index is 0, 1 row is enough
    if row_index==0:
        args['block'] = ((row_index, row_index+1), (0, size))
        return np.concatenate((np.zeros(1), compare(series, args)))

    # row index is last, 1 row is enough
    if row_index == size-1:
        args['block'] = ((0, size), (row_index, row_index + 1))
        return np.concatenate((compare(series, args), np.zeros(1)))

    args['block'] = ((0, row_index), (row_index, row_index + 1))
    row1 = compare(series, args)
    args['block'] = ((row_index, row_index+1), (row_index+1, size))
    row2 = compare(series, args)

    return np.concatenate((row1, np.zeros(1), row2))

