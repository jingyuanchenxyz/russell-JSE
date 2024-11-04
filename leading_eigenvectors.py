import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_filename = 'ret_matrix.csv'
ret_df = pd.read_csv(csv_filename, index_col=0, parse_dates=True)
n = 124

cov_matrix = ret_df @ ret_df.T / n
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

leading_eigv_2 = U[:,1].reshape(1,-1)
plt.figure(figsize=(10, 6))
plt.hist(leading_eigv_2[0], bins=20)
plt.xlabel('entries of leading_eigv_2')
plt.ylabel('frequency')
plt.title('Histogram of the Entries of the 2nd Leading Eigenvector')
plt.grid(True)
plt.show()

leading_eigv_3 = U[:,2].reshape(1,-1)
plt.figure(figsize=(10, 6))
plt.hist(leading_eigv_3[0], bins=20)
plt.xlabel('entries of leading_eigv_3')
plt.ylabel('frequency')
plt.title('Histogram of the Entries of the 3rn Leading Eigenvector')
plt.grid(True)
plt.show()

leading_eigv_4 = U[:,3].reshape(1,-1)
plt.figure(figsize=(10, 6))
plt.hist(leading_eigv_4[0], bins=20)
plt.xlabel('entries of leading_eigv_4')
plt.ylabel('frequency')
plt.title('Histogram of the Entries of the 4th Leading Eigenvector')
plt.grid(True)
plt.show()

