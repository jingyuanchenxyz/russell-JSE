import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
         '2022.7-2022.12',]

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

    es_cap = []
    for i in range(19, -1, -1):
        k = i * 100
        num_rows = p - k
        rows = ret.head(num_rows)
    
        cov_matrix = rows @ rows.T / n
        eigenvalues = np.linalg.eigvals(cov_matrix)
        es_cap.append(np.max(eigenvalues))

    plt.figure(figsize=(10, 6))
    plt.boxplot(es, positions=range(440, 2440, 100), widths=80, patch_artist=True)
    plt.plot(range(440, 2440, 100), es_cap, marker='x', label="Sorted by Cap", color='orange')
    plt.xticks(ticks=range(440, 2440, 100), rotation=45)
    plt.xlabel("p")
    plt.ylabel("Max eigenvalue")
    plt.title("Mean of the maximum eigenvalue of the n=126 covariance matrix for different p")
    plt.suptitle(f"Time period: {times[t]}", fontsize=10, color="gray")
    plt.grid(True)
    plt.savefig(f'eigenvalue_p{t}.png')
    plt.close()
