import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

n = 126
p = 3000
filenames = ['ret_matrix_19_1.csv',
             'ret_matrix_19_1.csv',
             'ret_matrix_19_7.csv',
             'ret_matrix_20_1.csv',
             'ret_matrix_20_7.csv',
             'ret_matrix_21_1.csv',
             'ret_matrix_21_7.csv',
             'ret_matrix_22_1.csv',
             'ret_matrix_22_7.csv',
             'ret_matrix_23_1.csv',
             'ret_matrix_23_7.csv']
times =['2020.7-2020.12',
        '2019.1-2019.6',
        '2019.7-2019.12',
        '2020.1-2020.6',
        '2020.7-2020.12',
        '2021.1-2021.6',
        '2021.7-2021.12',
        '2022.1-2022.6',
        '2022.7-2022.12',
        '2023.1-2023.6',
        '2023.7-2023.12']

top_evecs = []
evec_means_1 = []
evec_means_2 = []
evec_means_3 = []
evec_means_4 = []
evec_stds_1 = []
evec_stds_2 = []
evec_stds_3 = []
evec_stds_4 = []

for t, file in enumerate(filenames):
    ret = pd.read_csv(file, index_col=0)
    es = [[] for _ in range(30)]
    for m in range(30): # repeat m times
        np.random.seed(m)
        shuffled = ret.sample(frac=1).reset_index(drop=True)
        for i in range(29, -1, -1):
            k = i * 100
            num_rows = p - k
            rows = shuffled.head(num_rows)
            cov_matrix = rows.T.cov()
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues)
            es[i].append(np.max(eigenvalues))
            
    es.reverse()
    p_values = range(100, 3100, 100)
    n_samples = len(p_values)
    mean_es = [np.mean(vals) for vals in es]
    std_es = [np.std(vals) for vals in es]
    X = sm.add_constant(p_values)
    model = sm.OLS(mean_es, X).fit()
    intercept, slope = model.params

    es_cap = []
    for i in range(29, -1, -1):
        k = i * 100
        num_rows = p - k
        rows = ret.head(num_rows)
    
        cov_matrix = rows.T.cov()
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.real(eigenvalues)
        es_cap.append(np.max(eigenvalues))

    plt.figure(figsize=(10, 6))
    plt.boxplot(es, positions=p_values, widths=80, patch_artist=True)
    plt.plot(p_values, es_cap, marker='x', label="Sorted by Cap", color='orange')
    plt.xticks(ticks=p_values, rotation=45, fontsize=14)
    plt.xlabel("p", fontsize=18)
    plt.ylabel("Max eigenvalue", fontsize=18)
    plt.yticks(fontsize=14)
    plt.title("Maximum eigenvalue of the n=126 covariance matrix for different p", fontsize=14)
    plt.suptitle(f"Time period: {times[t]}", fontsize=18, color="gray")
    plt.grid(True)
    plt.savefig(f'eigenvalue_p_{times[t]}.png')
    plt.close()
    
    rows = ret.head(3000)
    data = rows.to_numpy()
    data_centered = data - np.mean(data, axis=1, keepdims=True)
    
    U, S, Vh = np.linalg.svd(data_centered)
    leading_eigv = U[:,0].reshape(1,-1)
    normalized_eigv = leading_eigv*p/np.sum(leading_eigv)
    variance = np.var(normalized_eigv)
    sigma = (slope * 252 / (1 + variance)) ** 0.5
    delta = (intercept * 252) ** 0.5
    print(f"Time: {times[t]}, sigma: {sigma}, delta: {delta}")

    eigenvalues = [x for x in eigenvalues if x >= 0.001]
    plt.hist(eigenvalues, bins=200)
    plt.title(f'Histogram of Eigenvalues {times[t]}')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.savefig(f'eigenvalue_hist_{times[t]}.png')
    plt.close()

    

    
