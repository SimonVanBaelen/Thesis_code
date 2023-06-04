# Thesis_code

This is the code used for my master's thesis, where an extendable version of Adaptive Cross Approximation was used to speed up the clustering of a growing amount of time series.
This implementation is an extension on the code given by Mathias Pede [2], which was published alongside [3]. In the experiments, the time series of the UCR archive were used [1].

#### Create a first ACA approximation
To create an ACA approximation, use the following code:

    from src.extendable_aca import ACA
    from src.cluster_problem import ClusterProblem
    cp = ClusterProblem(starting_series, "dtw", compare_args={"window": len(starting_series) - 1}, solved_matrix=None)
    ACA_approximation = ACA(cp, tolerance=0.05, max_rank=rank)
    
#### Extending previous ACA approximations
This ACA approximation can be extended using:

    ACA_approximation.extend(new_series, method=method)

As input for the 'method' parameter, the options are:
- For the skeleton update: "method1" or "skeleton update".
- For the tolerance-based update: "method2" or "tolerance-based update".
- For the adaptive update: "method3" or "adaptive update".
- For the exact update: "method4"  or "exact update".
- For the maximal update :"method5"  or"maximal update".

#### Get the current ACA approximation:
Use the following function:

    ACA_approximation.getApproximation()

## References

  [1]:  H. A. Dau, E. Keogh, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A.Ratanamahatana, Yanping, B. Hu, N. Begum, A. Bagnall, A. Mueen, G. Batista,and Hexagon-ML,
  “The ucr time series classification archive,” October 2018.https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.
   
  [2]: M. Pede. Fast-time-series-clustering, 2020. https://github.com/MathiasPede/Fast-Time-Series-Clustering Accessed: (October 23,2022).

  [3]: M. Pede. Snel clusteren van tijdreeksen via lage-rang benaderingen. Master’s
  thesis, Faculteit Ingenieurswetenschappen, KU Leuven, Leuven, Belgium, 2020.


## License

    Copyright 2022-2023 KU Leuven

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
