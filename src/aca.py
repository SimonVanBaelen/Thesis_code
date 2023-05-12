import numpy as np
import random as rnd
import logging
import copy

from math import sqrt, floor
from src.cluster_problem import ClusterProblem

logger = logging.getLogger("ftsc")

class ACA:
    def __init__(self, cp: ClusterProblem, tolerance=0.05, max_rank=None, start_index=None, seed=None,
                 start_indices=None, restart_deltas=None,restart_with_prev_pivots=False, start_samples=None):
        """
        Adaptive Cross Approximation for Symmetric Distance Matrices

        @param cp: Cluster Problem, includes the objects and a compare function
        @param tolerance: An estimated relative error of the resulting approximation
        @param max_rank: The maximal rank of the approximation (number of crosses)
        @param start_index: Optional first row to start the first cross
        @param seed: Optional seed to make algorithm deterministic

        @return: Approximation of the distance matrix of the cluster problem with an approximated relative error equal
        to the tolerance parameter.
        """

        if not 0.0 < tolerance < 1.0:
            logger.error("Opted tolerance not within [0.0,1.0] range")
        if not max_rank or max_rank > cp.size():
            logger.debug("Max rank set to maximum")
            max_rank = cp.size()
        if seed:
            rnd.seed(seed)

        self.cp = cp
        self.tolerance = tolerance
        self.max_rank = max_rank
        self.start_index = start_index
        self.start_indices = start_indices
        self.restart_with_prev_pivots = restart_with_prev_pivots
        self.restart_deltas = restart_deltas
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
        self.dtw_calculations = 0
        self.start_rank = len(self.rows)
        self.start_size = self.cp.size()

    def choose_starting_pivot(self, new_run, current_state=None):
        if new_run or len(self.rows) == 0:
            # Start random row
            if self.restart_with_prev_pivots:
                pivot_index = self.start_indices[0]
            elif self.start_index and 0 <= self.start_index <= self.cp.size():
                pivot_index = self.start_index
            else:
                pivot_index = rnd.randint(0, self.cp.size() - 1)
        else:
            pivot_index = self.choose_new_pivot(self.rows[-1], current_state)
        return pivot_index

    def choose_new_pivot(self, row, current_state):
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
                new_delta = self.restart_deltas[m]

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
                logger.debug("No improvement for 100 ranks, current rank: " + str(m))
                estimated_error = sqrt(current_aca_state["best remaining average"]) / sqrt(self.initial_average)
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
                    pivot_index = self.start_indices[m]
                    current_aca_state["prev_pivot"] = pivot_index
                except:
                    self.ACA_states.append(current_aca_state)
                    return best_m

            m += 1
            estimated_error = sqrt(current_aca_state["best remaining average"]) / sqrt(self.initial_average)

            if current_aca_state["stopcrit"]:
                logger.debug("stopcrit: Approximated error: " + str(estimated_error))
            else:
                logger.debug("Max rank " + str(self.max_rank) + "achieved, Approximated error: " + str(estimated_error))

            self.ACA_states.append(current_aca_state)

        return best_m

    def get_current_error(self):
        pass

    def getApproximation(self):
        results = self.calc_symmetric_matrix_approx(self.rows, self.deltas, self.current_rank)
        for i in self.full_dtw_rows:
            all_dtw = np.transpose(self.cp.sample_row(i)) #TODO
            results[:,i] = all_dtw
            results[i, :] = all_dtw
        return results


    def calc_symmetric_matrix_approx(self, rows, deltas, rank):
        rows_array = np.array(rows)[0:rank]
        deltas_array = np.array(deltas)[0:rank]
        cols = np.transpose(rows_array).copy()
        cols = np.divide(cols, deltas_array)
        result = np.matmul(cols, rows_array)
        np.fill_diagonal(result, 0)
        return result


    def generate_samples_student_distribution(self, error_margin=0.02):
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
        logger.debug("Sample size: " + str(amount_sampled))
        print(str(amount_sampled))
        return sample_indices, sample_values

    def extend(self, ts, dm=None, method="method1"):
        if self.max_rank >= self.cp.size():
            self.max_rank += 1
        self.cp.add_ts(ts, dm)
        if method == "method1":
            self.extend_prior_rows()
        elif method == "method2":
            self.extend_prior_rows()
            self.add_extra_samples()
            prev_rank = len(self.rows)
            self.current_rank = self.aca_symmetric_body(new_run=False)
            new_rows = len(self.rows) - prev_rank
            self.dtw_calculations += new_rows*self.cp.size() + self.amount_of_samples_per_row + prev_rank
        elif method == "method3":
            prev_rank = len(self.rows)
            self.extend_and_remove_prior_rows()
            self.add_extra_samples()
            self.current_rank = self.aca_symmetric_body(new_run=False)
            new_rows = len(self.rows) - prev_rank
            self.dtw_calculations += new_rows*self.cp.size() + self.amount_of_samples_per_row + prev_rank
        elif method == "method4":
            self.add_series_using_dm()
        elif method == "method5":
            self.extend_prior_rows()
            prev_rank = len(self.rows)
            self.current_rank = self.aca_symmetric_body(new_run=False, m5=True)
            new_rows = len(self.rows) - prev_rank
            self.dtw_calculations += new_rows * self.cp.size() + prev_rank




    def add_series_using_dm(self):
        next = self.cp.size()-1
        for i in range(len(self.rows)):
            self.rows[i] = np.append(self.rows[i], [0])
        self.full_dtw_rows.append(next)
        self.dtw_calculations += self.cp.size()

    def extend_prior_rows(self):
        for i in range(len(self.rows)):
            new_value = self.cp.sample(self.cp.size()-1, self.indices[i])
            approx = 0
            for j in range(i):
                approx += self.rows[j][self.indices[i]] * self.rows[j][-1] / self.deltas[j]
            new_value -= approx
            self.rows[i] = np.append(self.rows[i], [new_value])
            self.dtw_calculations += 1

    def extend_and_remove_prior_rows(self):
        for i in range(len(self.rows)):
            new_value = self.cp.sample(self.cp.size()-1, self.indices[i])
            approx = 0
            for j in range(i):
                approx += self.rows[j][self.indices[i]] * self.rows[j][-1] * (1.0 / self.deltas[j])
            new_value -= approx
            self.dtw_calculations += 1
            self.rows[i] = np.append(self.rows[i], [new_value])
            pivot = self.choose_new_pivot(self.rows[i],  self.ACA_states[i])
            if not pivot == self.indices[i]:
                self.rows = self.rows[:i]
                self.deltas = self.deltas[:i]
                self.indices = self.indices[:i]
                self.ACA_states = self.ACA_states[:i+1]
                return

    def get_DTW_calculations(self):
        start_calc = len(self.sample_indices) + self.start_rank*self.start_size
        return self.dtw_calculations + start_calc

    def add_extra_samples(self):
        amount = self.amount_of_samples_per_row
        new_sample_indices = np.zeros(shape=(amount, 2), dtype=int)
        new_sample_values = np.zeros(amount, dtype=float)
        # Take some more samples
        for i in range(self.amount_of_samples_per_row):
            x = self.cp.size()-1
            y = x
            while x == y:
                y = rnd.randint(0, self.cp.size() - 1)
            new_sample_indices[i, 0] = x
            new_sample_indices[i, 1] = y
            new_sample_values[i] = self.cp.sample(x, y)

        self.sample_indices = np.concatenate((self.sample_indices, new_sample_indices))
        self.sample_values = np.concatenate((self.sample_values, new_sample_values))

        # update all ACA states and reevaluate the new samples
        for m in range(0, len(self.ACA_states)):

            # reevaluate the new samples
            if m > 0:
                for j in range(len(new_sample_indices)):
                    x = new_sample_indices[j, 0]
                    y = new_sample_indices[j, 1]
                    new_sample_values[j] = new_sample_values[j] - self.rows[m-1][y] * self.rows[m-1][x]/ self.deltas[m-1]

            self.ACA_states[m]["sample_values"] = np.concatenate((self.ACA_states[m]["sample_values"], new_sample_values))

            if m > 0:
                # Update stopcriterium
                remaining_average = np.average(np.square(self.ACA_states[m]["sample_values"]))
                self.ACA_states[m]["stopcrit"] = (sqrt(remaining_average) < self.ACA_states[m]["max allowed relative error"])
                self.ACA_states[m]["best remaining average"] = remaining_average

                # Delete the samples on the pivot row from the restartable samples of the state
                pivot_indices_in_row_of_samples = np.where(self.sample_indices[:, 0] == self.indices[m-1])[0]
                pivot_indices_in_col_of_samples = np.where(self.sample_indices[:, 1] == self.indices[m-1])[0]
                pivot_indices_in_samples = np.concatenate((pivot_indices_in_row_of_samples, pivot_indices_in_col_of_samples))
                self.ACA_states[m]["deleted indices"] = np.concatenate((self.ACA_states[m]["deleted indices"], pivot_indices_in_samples), axis=0)
                self.ACA_states[m]["restartable samples"] = np.delete(self.ACA_states[m]["sample_values"], self.ACA_states[m]["deleted indices"], axis=0)
                self.ACA_states[m]["restartable indices"] = np.delete(self.sample_indices, self.ACA_states[m]["deleted indices"], axis=0)

                self.ACA_states[m]["max residu"] = np.max(np.abs(self.ACA_states[m]["restartable samples"]))
            else:
                self.initial_average = np.average(np.square(self.sample_values))
                self.ACA_states[m]["best remaining average"] = self.initial_average
                self.ACA_states[m]["max allowed relative error"] = sqrt(self.initial_average) * self.tolerance
                self.ACA_states[m]["restartable samples"] = np.copy(self.sample_values)
                self.ACA_states[m]["restartable indices"] = np.copy(self.sample_indices)
                self.ACA_states[m]["sample_values"] = np.copy(self.sample_values)