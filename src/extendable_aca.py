"""
###################################################################################################################
    Some of the functions/classes in this file are made by Mathias Pede and modified to be used in this work.
    The original code can be found in [1] and was published alongside [2]. The functions that were made
    by Mathias P. and modified were:
    1. __init__(...): Was originally a function called 'aca_symm' but was expanded to a class. The init-function was
    inspired by 'aca_symm'.
    2. choose_starting_pivot and choose_new_pivot: Were originally part of  'aca_symmetric_body'
    3. aca_symmetric_body: States were added to this function.
    4. calc_symmetric_matrix_approx
    5. generate_samples_student_distribution: Error margin was increased to 0.02

    [1]: M. Pede. Fast-time-series-clustering, 2020.
    https://github.com/MathiasPede/Fast-Time-Series-Clustering Accessed: (October 23,2022).

    [2]: M. Pede. Snel clusteren van tijdreeksen via lage-rang benaderingen. Master’s
    thesis, Faculteit Ingenieurswetenschappen, KU Leuven, Leuven, Belgium, 2020.
###################################################################################################################
"""

import numpy as np
import random as rnd
import copy
from math import sqrt
from src.cluster_problem import ClusterProblem

class ACA:
    def __init__(self, cp: ClusterProblem, tolerance=0.05, max_rank=None, start_index=None, seed=None,
                 given_indices=None, given_deltas=None,restart_with_prev_pivots=False, start_samples=None):
        """
        Creates an Adaptive Cross Approximation objects, which can be extended using five different extension methods.
        
        :param cp: A cluster problem which contains the timeseries or a solved distancematrix
        :param tolerance: The tolerance for the tolerated error
        :param max_rank: The maximum rank of the ACA approximation.
        :param start_index: The start index of the approximation.
        :param seed: The seed.
        :param given_indices: Force an ACA approximation to use the given indices
        :param given_deltas: Force an ACA approximation to use the given deltas (given_indices should not be None)
        :param restart_with_prev_pivots: Boolean that when enabled, forces the given_indices and given_deltas
        :param start_samples: Start with a given collection of samples.
        """

        if not max_rank or max_rank > cp.size():
            max_rank = cp.size()
        if seed:
            rnd.seed(seed)

        self.cp = cp
        self.tolerance = tolerance
        self.max_rank = max_rank
        self.start_index = start_index
        self.given_indices = given_indices
        self.restart_with_prev_pivots = restart_with_prev_pivots
        self.given_deltas = given_deltas
        if start_samples is None:
            self.sample_indices, self.sample_values = self.generate_samples_student_distribution()
        else:
            self.sample_indices = np.copy(start_samples[0])
            self.sample_values = np.copy(start_samples[1])
        self.amount_of_samples_per_row = int(len(self.sample_indices)/cp.size())
        self.initial_average = np.average(np.square(self.sample_values))
        self.rows = []
        self.deltas = []
        self.indices = []
        ACA_state_start = {
            "best remaining average":  self.initial_average,
            "max allowed relative error": sqrt(self.initial_average) * self.tolerance,
            "stopcrit": False,
            "max residu": 0.0,
            "deleted indices": np.array([], dtype=int),
            "restartable samples": np.copy(self.sample_values),
            "restartable indices": np.copy(self.sample_indices),
            "n_rows": 0,
            "sample_values": np.copy(self.sample_values),
            "prev_pivot": rnd.randint(0, self.cp.size() - 1)
        }
        self.ACA_states = [ACA_state_start]
        self.current_rank = self.aca_symmetric_body()
        self.full_dtw_rows = []
        self.dtw_calculations = len(self.sample_indices)
        self.start_rank = len(self.rows)
        self.start_size = self.cp.size()

    def choose_starting_pivot(self, new_run, current_state=None):
        """
        Function that chooses a starting pivot for the ACA algorithm.
        :param new_run: Boolean that signifies if this is a new run or not.
        :param current_state: Needed if new_run = False.
        """
        if new_run or len(self.rows) == 0:
            # Start random row
            if self.restart_with_prev_pivots:
                pivot_index = self.given_indices[0]
            elif self.start_index and 0 <= self.start_index <= self.cp.size():
                pivot_index = self.start_index
            else:
                pivot_index = rnd.randint(0, self.cp.size() - 1)
        else:
            pivot_index = self.choose_new_pivot(self.rows[-1], current_state)
        return pivot_index

    def choose_new_pivot(self, row, current_state):
        """
        Chooses a new pivot in a given row and state.
        """
        new_row_abs = np.abs(row)
        row_without_already_sampled_indices = np.delete(new_row_abs, self.indices, axis=0)
        new_max = np.max(row_without_already_sampled_indices)
        tmp = np.where(new_row_abs == new_max)
        pivot_index = tmp[0][0]

        # Check whether the max of the row is smaller than the max residu from the samples, if so, switch
        if abs(new_max) < current_state["max residu"] - 0.001:
            # Switch to the pivot to the sample with the largest remaining value
            index_sample_max = np.where(np.abs(current_state["restartable samples"]) == current_state["max residu"])[0][0]
            pivot_index = current_state["restartable indices"][index_sample_max][0]

        return pivot_index


    def aca_symmetric_body(self, iters_no_improvement=100, new_run=True, m5=False):
        """
        Runs the ACA algorithm starting from the last ACA state, if the last ACA state is self.ACA_states[0], new_run
        should be enabled. If m5=True one iteration of the ACA algorithm is forced, is used for the maximal update.
        """
        m = len(self.rows)
        best_m = len(self.rows)
        current_aca_state = copy.deepcopy(self.ACA_states[-1])
        pivot_index = self.choose_starting_pivot(new_run, current_state=current_aca_state)
        if m5:
            current_aca_state["stopcrit"] = False
        while m < self.max_rank and not current_aca_state["stopcrit"]:
            current_aca_state = copy.deepcopy(self.ACA_states[-1])
            # Calculate the current approximation for row of pivot
            approx = np.zeros(self.cp.size())
            for i in range(m):
                approx = np.add(approx, self.rows[i] * self.rows[i][pivot_index] * (1.0 / self.deltas[i]))

            # Find new w vector
            new_row = np.subtract(self.cp.sample_row(pivot_index), approx)
            # Find delta at the pivot index
            if not self.restart_with_prev_pivots:
                new_delta = new_row[pivot_index]
            else:
                new_delta = self.given_deltas[m]

             # If delta is zero, substitute by max value in the w vector
            if new_delta == 0:
                new_max = np.max(np.abs(new_row))
                # If the maximum is also 0 (row is perfectly approximated) take a new pivot from the samples
                if new_max == 0.0:
                    index_sample_max = np.where(np.abs(current_aca_state["restartable samples"]) == current_aca_state["max residu"])[0][0]
                    pivot_index = current_aca_state["restartable indices"][index_sample_max][0]
                    continue
                new_delta = new_max

            # Add the cross
            self.indices.append(pivot_index)
            self.rows.append(new_row)
            self.deltas.append(new_delta)

            current_aca_state["n_rows"] = len(self.rows)
            # Reevaluate the samples
            for j in range(len(current_aca_state["sample_values"])):
                x = self.sample_indices[j, 0]
                y = self.sample_indices[j, 1]

                current_aca_state["sample_values"][j] = current_aca_state["sample_values"][j] - (1.0 / self.deltas[m]) * self.rows[m][y] * self.rows[m][x]

            # Estimate the frobenius norm and check stop criterion
            remaining_average = np.average(np.square(current_aca_state["sample_values"]))
            current_aca_state["stopcrit"] = (sqrt(remaining_average) < current_aca_state["max allowed relative error"])

            # If average entry is lower the previous best, continue, otherwise check whether no improvement for x iters
            if remaining_average < current_aca_state["best remaining average"]:
                current_aca_state["best remaining average"] = remaining_average
                best_m = m
            elif m > best_m + iters_no_improvement:
                self.ACA_states.append(current_aca_state)
                return best_m

            # Delete the samples on the pivot row from the restartable samples
            pivot_indices_in_row_of_samples = np.where(self.sample_indices[:, 0] == pivot_index)[0]
            pivot_indices_in_col_of_samples = np.where(self.sample_indices[:, 1] == pivot_index)[0]
            pivot_indices_in_samples = np.concatenate((pivot_indices_in_row_of_samples, pivot_indices_in_col_of_samples))
            if current_aca_state["deleted indices"].size == 0:
                current_aca_state["deleted indices"] = pivot_indices_in_samples
            else:
                current_aca_state["deleted indices"] = np.concatenate((current_aca_state["deleted indices"], pivot_indices_in_samples), axis=0)

            current_aca_state["restartable samples"] = np.delete(current_aca_state["sample_values"], current_aca_state["deleted indices"], axis=0)
            current_aca_state["restartable indices"] = np.delete(self.sample_indices, current_aca_state["deleted indices"], axis=0)


            # Find the maximum error on the samples
            if current_aca_state["restartable samples"].size == 0:
                current_aca_state["max residu"] = 0
            else:
                current_aca_state["max residu"] = np.max(np.abs(current_aca_state["restartable samples"]))


            # Choose a new pivot
            if not self.restart_with_prev_pivots:
                pivot_index = self.choose_new_pivot(new_row, current_aca_state)
                current_aca_state["prev_pivot"] = pivot_index
            else:
                try:
                    pivot_index = self.given_indices[m]
                    current_aca_state["prev_pivot"] = pivot_index
                except:
                    self.ACA_states.append(current_aca_state)
                    return best_m

            m += 1
            estimated_error = sqrt(current_aca_state["best remaining average"]) / sqrt(self.initial_average)

            self.ACA_states.append(current_aca_state)
            if m5:
                return best_m
        return best_m

    def getApproximation(self):
        """
        Function that returns the approximation and fills in the rows calculated in the exact update.
        """
        results = self.calc_symmetric_matrix_approx(self.rows, self.deltas, self.current_rank)
        for i in self.full_dtw_rows:
            all_dtw = np.transpose(self.cp.sample_row(i))
            results[:,i] = all_dtw
            results[i, :] = all_dtw
        return results


    def calc_symmetric_matrix_approx(self, rows, deltas, rank):
        """
        Calculates the ACA approximation.
        """
        rows_array = np.array(rows)[0:rank]
        deltas_array = np.array(deltas)[0:rank]
        cols = np.transpose(rows_array).copy()
        cols = np.divide(cols, deltas_array)
        result = np.matmul(cols, rows_array)
        np.fill_diagonal(result, 0)
        return result


    def generate_samples_student_distribution(self, error_margin=0.02):
        """
        Generates the samples for ACA stopcrit.
        """
        amount_sampled = 0
        t = 3.39
        tolerance = np.infty
        size = self.cp.size()
        sample_indices = None
        sample_values = None
        while tolerance > error_margin or amount_sampled < 2 * self.cp.size():
            iteration_indices = np.zeros(shape=(size, 2), dtype=int)
            iteration_values = np.zeros(size, dtype=float)

            # Take size more samples
            for i in range(size):
                x = i
                y = i
                while x == y:
                    y = rnd.randint(0, size - 1)
                iteration_indices[i, 0] = x
                iteration_indices[i, 1] = y
                iteration_values[i] = self.cp.sample(x, y)

            # Add the samples to the already sampled values
            if amount_sampled == 0:
                sample_indices = iteration_indices
                sample_values = iteration_values
            else:
                sample_indices = np.concatenate((sample_indices, iteration_indices))
                sample_values = np.concatenate((sample_values, iteration_values))

            # If sample size becomes too large, stop
            amount_sampled += size
            if amount_sampled > self.cp.size() * self.cp.size():
                break

            # Calculate the new current error margin
            squared = np.square(sample_values)
            average_so_far = np.mean(squared)
            std_so_far = np.std(squared)
            tolerance = (t * std_so_far) / (sqrt(amount_sampled) * average_so_far)
        return sample_indices, sample_values

    def extend(self, timeseries, solved_matrix=None, method="method1"):
        """
        A function that extends a given ACA approximation given an update method.
        """
        if self.max_rank >= self.cp.size():
            self.max_rank += 1
        start_index = self.cp.size()
        self.cp.add_timeseries(timeseries, solved_matrix)
        end_index = self.cp.size()
        if method == "method1" or method == "skeleton update":
            self.do_skeleton_update(start_index, end_index)
        elif method == "method2" or method == "tolerance-based update":
            self.do_tolerance_based_additive_update(start_index, end_index)
        elif method == "method3" or method == "adaptive update":
            self.do_adaptive_update(start_index, end_index)
        elif method == "method4"  or method == "exact update":
            self.do_exact_additive_update(start_index, end_index)
        elif method == "method5"  or method == "maximal update":
            self.do_maximal_additive_update(timeseries, start_index, end_index)

    def do_exact_additive_update(self, start_index, end_index):
        """
        Does an exact update.
        """
        for next in range(start_index, end_index):
            for i in range(len(self.rows)):
                self.rows[i] = np.append(self.rows[i], [0])
            self.full_dtw_rows.append(next)
        self.dtw_calculations += self.cp.size() * (end_index-start_index)

    def do_tolerance_based_additive_update(self, start_index, end_index):
        """
        Does an tolerance-based update.
        """
        self.do_skeleton_update(start_index, end_index)
        self.add_extra_samples_and_update_states(start_index, end_index)
        prev_rank = len(self.rows)
        self.current_rank = self.aca_symmetric_body(new_run=False)
        new_rows = len(self.rows) - prev_rank
        self.dtw_calculations += new_rows * self.cp.size() + (end_index-start_index)*self.amount_of_samples_per_row

    def do_skeleton_update(self, start_index, end_index):
        """
        Does an skeleton update.
        """
        for i in range(len(self.rows)):
            new_values = []
            for m in range(start_index, end_index):
                new_value = self.cp.sample(m, self.indices[i])
                approx = 0
                for j in range(i):
                    approx += self.rows[j][self.indices[i]] * self.rows[j][m] / self.deltas[j]
                new_value -= approx
                new_values.append(new_value)
            self.rows[i] = np.append(self.rows[i], new_values)
        self.dtw_calculations += (end_index-start_index)*len(self.rows)

    def do_maximal_additive_update(self, timeseries, start_index, end_index):
        """
        Does an maximal update.
        """
        self.do_skeleton_update(start_index, end_index)
        for _ in timeseries:
            self.current_rank = self.aca_symmetric_body(new_run=False, m5=True)
        self.dtw_calculations += (end_index-start_index) * self.cp.size() + (end_index-start_index)*len(self.rows)

    def do_adaptive_update(self, start_index, end_index):
        """
        Does an adaptive update.
        """
        prev_rank = len(self.rows)
        self.extend_and_remove_prior_rows(start_index,end_index)
        removed = abs(len(self.rows) - prev_rank)
        self.current_rank = self.aca_symmetric_body(new_run=False)
        new_rows = abs(len(self.rows) - removed)
        self.dtw_calculations += new_rows * self.cp.size() + (end_index-start_index)*(self.amount_of_samples_per_row + abs(prev_rank-removed))

    def extend_and_remove_prior_rows(self, start_index, end_index):
        """
        Step 1 of the adaptive update.
        """
        new_sample_values, new_sample_indices = self.find_new_samples_for_ts(start_index)
        for m in range(start_index+1, end_index):
            tmp_sv, tmp_si = self.find_new_samples_for_ts(m)
            new_sample_indices = np.concatenate((new_sample_indices, tmp_si))
            new_sample_values = np.concatenate((new_sample_values, tmp_sv))

        for i in range(len(self.rows)):
            self.update_state_new_samples(i, self.ACA_states[i], new_sample_values, new_sample_indices)
            for m in range(start_index, end_index):
                new_value = self.cp.sample(m, self.indices[i])
                approx = 0
                for j in range(i):
                    approx += self.rows[j][self.indices[i]] * self.rows[j][m] * (1.0 / self.deltas[j])
                new_value -= approx
                self.rows[i] = np.append(self.rows[i], [new_value])
                pivot = self.choose_new_pivot(self.rows[i],  self.ACA_states[i])
                if not pivot == self.indices[i]:
                    self.rows = self.rows[:i]
                    self.deltas = self.deltas[:i]
                    self.indices = self.indices[:i]
                    self.ACA_states = self.ACA_states[:i+1]
                    return


    def get_DTW_calculations(self):
        """
        Returns the amount of DTW-calculations done by the current ACA algorithm and possible updates.
        """
        start_calc = self.start_rank*self.start_size
        return self.dtw_calculations + start_calc

    def add_extra_samples_and_update_states(self, start_index, end_index):
        """
        Finds new samples and updates previous ACA states.
        """
        for m in range(start_index, end_index):
            new_sample_values, new_sample_indices = self.find_new_samples_for_ts(m)
            for index_state, state in zip(range(len(self.ACA_states)), self.ACA_states):
                self.update_state_new_samples(index_state, state, new_sample_values, new_sample_indices)

    def find_new_samples_for_ts(self, index):
        amount = self.amount_of_samples_per_row
        new_sample_indices = np.zeros(shape=(amount, 2), dtype=int)
        new_sample_values = np.zeros(amount, dtype=float)
        # Take some more samples
        for i in range(amount):
            x = index
            y = x
            while x == y and y not in self.indices:
                y = rnd.randint(0, index)
            new_sample_indices[i, 0] = x
            new_sample_indices[i, 1] = y
            new_sample_values[i] = self.cp.sample(x, y)

        self.sample_indices = np.concatenate((self.sample_indices, new_sample_indices))
        self.sample_values = np.concatenate((self.sample_values, new_sample_values))
        return new_sample_values, new_sample_indices

    def update_state_new_samples(self, m, state, new_sample_values, new_sample_indices):
        # reevaluate the new samples
        if m > 0:
            for j in range(len(new_sample_indices)):
                x = new_sample_indices[j, 0]
                y = new_sample_indices[j, 1]
                new_sample_values[j] = new_sample_values[j] - self.rows[m - 1][y] * self.rows[m - 1][x] / \
                                       self.deltas[m - 1]

        state["sample_values"] = np.concatenate(
            (state["sample_values"], new_sample_values))

        if m > 0:
            # Update stopcriterium
            remaining_average = np.average(np.square(state["sample_values"]))
            state["stopcrit"] = (
                        sqrt(remaining_average) < state["max allowed relative error"])
            state["best remaining average"] = remaining_average

            # Delete the samples on the pivot row from the restartable samples of the state
            pivot_indices_in_row_of_samples = np.where(self.sample_indices[:, 0] == self.indices[m - 1])[0]
            pivot_indices_in_col_of_samples = np.where(self.sample_indices[:, 1] == self.indices[m - 1])[0]
            pivot_indices_in_samples = np.concatenate((pivot_indices_in_row_of_samples, pivot_indices_in_col_of_samples))
            state["deleted indices"] = np.concatenate((state["deleted indices"], pivot_indices_in_samples), axis=0)
            state["restartable samples"] = np.delete(state["sample_values"], state["deleted indices"], axis=0)
            state["restartable indices"] = np.delete(self.sample_indices,state["deleted indices"], axis=0)

            self.ACA_states[m]["max residu"] = np.max(np.abs(state["restartable samples"]))
        else:
            self.initial_average = np.average(np.square(self.sample_values))
            state["best remaining average"] = self.initial_average
            state["max allowed relative error"] = sqrt(self.initial_average) * self.tolerance
            state["restartable samples"] = np.copy(self.sample_values)
            state["restartable indices"] = np.copy(self.sample_indices)
            state["sample_values"] = np.copy(self.sample_values)