import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n = 124
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
ev_1 = []
ev_2 = []
ev_3 = []
ev_4 = []
for file in filenames:
    ret = pd.read_csv(file, index_col=0, parse_dates=True)
    cov = ret.T @ ret / n
    evs, _ = np.linalg.eig(cov.values)
    evs = np.sort(evs)
    ev_1.append(evs[-1])
    ev_2.append(evs[-2])
    ev_3.append(evs[-3])
    ev_4.append(evs[-4])
    
plt.figure(figsize=(10, 6))
plt.plot(times, ev_1, marker='o', label='Largest Eigenvalue (ev_1)')
plt.plot(times, ev_2, marker='o', label='2nd Largest Eigenvalue (ev_2)')
plt.plot(times, ev_3, marker='o', label='3rd Largest Eigenvalue (ev_3)')
plt.plot(times, ev_4, marker='o', label='4th Largest Eigenvalue (ev_4)')

# Labeling
plt.xlabel('Time Periods')
plt.ylabel('Eigenvalue')
plt.title('Top 4 Eigenvalues for Each Time Period')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

top_evecs = []
for file in filenames:
    ret = pd.read_csv(file, index_col=0, parse_dates=True)
    U, S, Vh = np.linalg.svd(ret)
    top_evec = U[:,1].reshape(1,-1)[0]
    normalized_evec = top_evec / np.mean(top_evec)
    top_evecs.append(normalized_evec)

plt.figure(figsize=(10, 6))
plt.boxplot(top_evecs, labels=times, showmeans=True)
plt.xlabel('Time Periods')
plt.ylabel('Normalized Eigenvector Entries')
plt.title('Box Plot of Second Leading Eigenvectors Across Time Periods')
plt.grid(True)
plt.ylim(-100,100)
plt.tight_layout()
plt.show()
quit()

    

cov_matrix = ret_df.T @ ret_df / n
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix.values)
U, S, Vh = np.linalg.svd(ret_df)

leading_eigv_1 = U[:,0].reshape(1,-1)
plt.figure(figsize=(10, 6))
plt.hist(leading_eigv_1[0], bins=20)
plt.xlabel('entries of leading_eigv_1')
plt.ylabel('frequency')
plt.title('Histogram of the Entries of the 1st Leading Eigenvector')
plt.grid(True)
plt.show()
