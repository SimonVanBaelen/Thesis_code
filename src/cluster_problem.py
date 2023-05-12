"""
    MATHIAS PEDE

    Class ClusterProblem

    Used for storing the distance/similarity matrix. Starts with uninitialized elements

    Can be sampled

"""

import numpy as np
import math
from .distance_functions import compute_distance, compute_row


class ClusterProblem:
    def __init__(self, series, compare, compare_args=None, solved_matrix=None):
        """
        Creates a cluster problem objects, which contains the data objects. A distance matrix is created with initially
        NaN values and entries can be computed based on index using the 'compare' function
        :param series: The data objects (array of time series, usually with first index the class number)
        :param compare: Distance function
        :param solved_matrix: Potentially add an already solved matrix to speed up
        """
        self.series = series
        self.compare = compare
        self.compare_args = compare_args
        self.matrix = np.empty((len(series), len(series)), dtype=np.float_)
        self.matrix[:] = np.NaN
        self.solved_matrix = solved_matrix

    def size(self):
        return len(self.series)

    def get_time_serie(self, x):
        return self.series[x]

    def read(self, x, y):
        return self.matrix[x][y]

    def write(self, x, y, val):
        self.matrix[x][y] = val
        self.matrix[y][x] = val

    def sample(self, x, y):
        # if the element has not yet been calculated, calculate
        value = self.read(x, y)
        if math.isnan(value):
            if self.solved_matrix is not None:
                value = self.solved_matrix[x][y]
            else:
                value = compute_distance(self.get_time_serie(x), self.get_time_serie(y),
                                         self.compare, self.compare_args)
            self.write(x, y, value)
        return value

    def sample_row(self, row_index):
        if self.solved_matrix is not None:
            self.matrix[row_index, :] = self.solved_matrix[row_index, :]
        else:
            if np.isnan(self.matrix[row_index, :]).any():
                row = compute_row(self.series, row_index, self.compare, self.compare_args)
                self.matrix[row_index, :] = row
                self.matrix[:, row_index] = row
        return self.matrix[row_index, :]

    def add_ts(self, ts, sm=None):
        self.series = np.vstack([self.series, [ts]])
        self.matrix = np.append(self.matrix, [np.full(len(self.matrix), np.nan)], 0)
        self.matrix = np.append(self.matrix, np.transpose([np.full(len(self.matrix), np.nan)]), 1)

        if not sm is None:
            self.solved_matrix = sm