import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def get_data_with_dates(end_date='2021-12-31', lookback_days=126):
    """Load data with explicit date range check"""
    csv_filename = 'ret_matrix.csv'
    ret_df = pd.read_csv(csv_filename, index_col=0)
    
    # Ensure index is datetime
    ret_df.index = pd.to_datetime(ret_df.index)
    
    # Sort index to ensure chronological order
    ret_df = ret_df.sort_index()
    
    # Get end date data
    end_date = pd.to_datetime(end_date)
    start_date = end_date - pd.Timedelta(days=lookback_days)
    
    # Select date range
    mask = (ret_df.index <= end_date) & (ret_df.index > start_date)
    ret_df_subset = ret_df[mask]
    
    print(f"\nData Range:")
    print(f"Start Date: {ret_df_subset.index.min()}")
    print(f"End Date: {ret_df_subset.index.max()}")
    print(f"Number of trading days: {len(ret_df_subset)}")
    
    return ret_df_subset.values.T  # Return as p x n matrix

def compute_mp_bounds(Y):
    """
    Compute Marchenko-Pastur distribution bounds
    Y: data matrix (p x n)
    """
    p, n = Y.shape
    q = n/p
    
    # Convert returns to percentage if needed
    if np.mean(np.abs(Y)) < 0.1:
        Y = Y * 100
    
    # Compute RMS volatility properly
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean(np.var(Y_centered, axis=1)))
    
    # MP bounds
    lambda_plus = sigma**2 * (1 + 1/np.sqrt(q))**2
    lambda_minus = sigma**2 * (1 - 1/np.sqrt(q))**2
    
    return lambda_plus, lambda_minus, q

def plot_eigenvalue_spectrum(eigenvalues, lambda_plus, lambda_minus, data_shape):
    """Plot eigenvalue spectrum with MP bounds"""
    p, n = data_shape
    q = n/p
    
    plt.figure(figsize=(12, 8))
    
    valid_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    plt.plot(range(1, len(valid_eigenvalues) + 1), valid_eigenvalues, 'bo-', 
             label='Observed Eigenvalues', alpha=0.6, markersize=3)
    
    plt.axhline(y=lambda_plus, color='r', linestyle='--', 
                label=f'MP Upper Bound: {lambda_plus:.2f}')
    plt.axhline(y=lambda_minus, color='r', linestyle='--',
                label=f'MP Lower Bound: {lambda_minus:.2f}')
    
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Spectrum with Marchenko-Pastur Bounds')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Print diagnostics
    print("\nEigenvalue Analysis:")
    print(f"Number of assets (p): {p}")
    print(f"Number of time periods (n): {n}")
    print(f"Dimension ratio (n/p): {q:.3f}")
    print(f"MP upper bound: {lambda_plus:.2f}")
    print(f"MP lower bound: {lambda_minus:.2f}")
    
    plt.show()

def plot_eigenvector_histogram(eigvec, index):
    """Plot histogram of eigenvector entries"""
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    sns.histplot(eigvec, bins=50, kde=True)
    
    plt.xlabel(f'Entries of Eigenvector {index}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Eigenvector {index} Entries')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Mean: {np.mean(eigvec):.3f}\n'
    stats_text += f'Std: {np.std(eigvec):.3f}\n'
    stats_text += f'Skew: {stats.skew(eigvec):.3f}\n'
    stats_text += f'Kurt: {stats.kurtosis(eigvec):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    plt.show()

def analyze_eigenvectors(eigenvectors, n_components=4):
    """Analyze and plot top eigenvectors"""
    for i in range(n_components):
        eigvec = eigenvectors[:, i]
        # Ensure positive mean
        if np.mean(eigvec) < 0:
            eigvec = -eigvec
        plot_eigenvector_histogram(eigvec, i+1)

def main():
    # Parameters
    end_date = '2021-12-31'
    lookback_days = 126
    n_components = 4
    
    # Load data with date verification
    returns = get_data_with_dates(end_date, lookback_days)
    data_shape = returns.shape
    
    # Make sure returns are in percentage form
    if np.mean(np.abs(returns)) < 0.1:
        returns = returns * 100
    
    # Center the returns
    returns_centered = returns - np.mean(returns, axis=1, keepdims=True)
    
    # Compute covariance matrix
    n = returns_centered.shape[1]
    cov_matrix = (returns_centered @ returns_centered.T) / n
    
    # Get eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.real(eigenvalues)  # Ensure real values
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute MP bounds
    lambda_plus, lambda_minus, q = compute_mp_bounds(returns_centered)
    
    # Plot spectrum
    plot_eigenvalue_spectrum(eigenvalues, lambda_plus, lambda_minus, data_shape)
    
    # Count significant factors
    significant = eigenvalues > lambda_plus
    n_factors = np.sum(significant)
    print(f"\nNumber of eigenvalues above MP bound: {n_factors}")
    
    # Print largest eigenvalues
    print("\nLargest eigenvalues:")
    for i in range(min(5, len(eigenvalues))):
        ratio = eigenvalues[i] / lambda_plus
        print(f"λ_{i+1}: {eigenvalues[i]:.2f} ({ratio:.1f}x MP bound)")
    
    # Analyze eigenvectors
    analyze_eigenvectors(eigenvectors, n_components)
    
    # Print variance explained
    total_var = np.sum(eigenvalues)
    var_explained = np.cumsum(eigenvalues) / total_var * 100
    print("\nVariance explained by top factors:")
    for i in range(min(5, len(eigenvalues))):
        print(f"Top {i+1} factors: {var_explained[i]:.1f}%")

if __name__ == "__main__":
    main()