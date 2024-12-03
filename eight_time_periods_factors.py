import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

n = 124
p = 2340
filenames = ['ret_matrix_19_1.csv',
             'ret_matrix_19_7.csv',
             'ret_matrix_20_1.csv',
             'ret_matrix_20_7.csv',
             'ret_matrix_21_1.csv',
             'ret_matrix_21_7.csv',
             'ret_matrix_22_1.csv',
             'ret_matrix_22_7.csv']
times = ['2019.1-2019.6',
         '2019.7-2019.12',
         '2020.1-2020.6',
         '2020.7-2020.12',
         '2021.1-2021.6',
         '2021.7-2021.12',
         '2022.1-2022.6',
         '2022.7-2022.12']

for t, file in enumerate(filenames):
    ret = pd.read_csv(file, index_col=0)
    es = [[] for _ in range(20)]
    for m in range(100):
        np.random.seed(m)
        shuffled = ret.sample(frac=1).reset_index(drop=True)
        for i in range(19, -1, -1):
            k = i * 100
            num_rows = p - k
            rows = shuffled.head(num_rows)
            cov_matrix = rows.T @ rows / n
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix.values)
            es[i].append(np.max(eigenvalues))
    es.reverse()
    p_values = range(440, 2440, 100)
    n_samples = len(p_values)
    mean_es = [np.mean(vals) for vals in es]
    std_es = [np.std(vals) for vals in es]
    X = sm.add_constant(p_values)
    model = sm.OLS(mean_es, X).fit()
    intercept, slope = model.params
    #print(f"Intercept: {intercept}")
    #print(f"Slope: {slope}")
    #print(model.summary())
    U, S, Vh = np.linalg.svd(ret)
    leading_eigv = U[:,0].reshape(1,-1)
    normalized_eigv = leading_eigv*p/np.sum(leading_eigv)
    variance = np.var(normalized_eigv)
    sigma = (slope * 252 / (1 + variance)) ** 0.5
    delta = (intercept * 252) ** 0.5
    print(f"Time: {times[t]}, sigma: {sigma}, delta: {delta}")
