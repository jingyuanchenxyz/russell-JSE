import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

csv_filename = 'ret_matrix.csv'
ret_df = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
p = 2340

es = []
for i in range(19, -1, -1):
    max_es = []
    for m in range (100):
        np.random.seed(m)
        k = i * 100
        num_rows = p - k
        selected_rows = np.random.choice(p, num_rows, replace=False)
        new = ret_df.iloc[selected_rows]
        cov_matrix = new.cov()
        eigenvalues = np.linalg.eigvals(cov_matrix)
        max_es.append(np.max(eigenvalues))

    es.append(max_es)

mean_es = [np.mean(vals) for vals in es]

es_cap = []
print(ret_df.head(1))
for i in range(19, -1, -1):
    k = i * 100
    num_rows = p - k
    rows = ret_df.head(num_rows)
    
    cov_matrix = rows.cov()
    eigenvalues = np.linalg.eigvals(cov_matrix)
    es_cap.append(np.max(eigenvalues))

plt.figure(figsize=(10, 6))
plt.plot(range(340, 2340, 100), mean_es, marker='o', label="Random")
plt.plot(range(340, 2340, 100), es_cap, marker='x', label="Sorted by Cap")
plt.xlabel("p - k")
plt.ylabel("Mean of the max eigenvalue")
plt.title("The mean of the maximum eigenvalue of the covariance matrix for each k")
plt.grid(True)
plt.show()

k_values = range(340, 2340, 100)
n_samples = len(k_values)

plt.figure(figsize=(10, 6))
plt.boxplot(es, positions=np.arange(len(k_values)), widths=0.6)
plt.xticks(np.arange(len(k_values)), k_values)
plt.xlabel("p - k")
plt.ylabel("Max Eigenvalue")
plt.title("Box plot of the maximum eigenvalue of the covariance matrix for each k")
plt.grid(True)
plt.show()

cov_matrix = ret_df.cov()
cov_filename = 'cov_matrix.csv'
cov_matrix.to_csv(cov_filename, encoding='utf-8')








