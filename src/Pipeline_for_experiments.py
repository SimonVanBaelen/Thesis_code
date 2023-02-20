"""
Should contain basic framework/pipeline for experiment with incremental addition of time series:

# Input: 1. Data being used
         2. Starting point of data (for incremental addition)
         3. ACA (variant) being tested
         4. Ground-truth
         5.(optional) Given previous ACA approximation.
         6. All to-be-tested clustering-algorithms
         7. Possible clustering evaluation criteria

# Output (Results): 1. Clusterings found with ACA variation.
                    2. Evaluation of a clustering for an evaluation criteria.
                    3. Evaluation of each clustering for each evaluation criteria.
                    4. Possible visualisations for results
"""
