import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os

def get_returns_data(end_date, lookback_days, p_stocks, conn):
    """Gets returns matrix Y (p x n) for top p stocks by market cap"""
    print(f"\nGetting data for p={p_stocks}")
    
    # Get trading days
    days_query = """
    SELECT DISTINCT DlyCalDt
    FROM russell3000
    WHERE DlyCalDt <= ?
    ORDER BY DlyCalDt DESC
    LIMIT ?
    """
    trading_days = pd.read_sql_query(days_query, conn, params=(end_date, lookback_days))
    trading_days = sorted(trading_days['DlyCalDt'].tolist())
    print(f"Found {len(trading_days)} trading days")
    
    if len(trading_days) < lookback_days:
        raise ValueError(f"Only found {len(trading_days)} trading days")
    
    # Get returns data
    days_str = "','".join(trading_days)
    returns_query = f"""
    SELECT DISTINCT DlyCalDt, Ticker, DlyRet, DlyCap
    FROM russell3000
    WHERE DlyCalDt IN ('{days_str}')
    ORDER BY DlyCalDt, Ticker
    """
    
    returns_df = pd.read_sql_query(returns_query, conn)
    returns_df['DlyRet'] = pd.to_numeric(returns_df['DlyRet'], errors='coerce')
    returns_df['DlyCap'] = pd.to_numeric(returns_df['DlyCap'], errors='coerce')
    returns_df = returns_df.drop_duplicates(subset=['DlyCalDt', 'Ticker'], keep='first')
    
    # Create returns matrix
    Y_full = returns_df.pivot(index='DlyCalDt', columns='Ticker', values='DlyRet')
    Y_full = Y_full.dropna(axis=1)
    
    # Get top p stocks by market cap
    last_day_caps = returns_df[returns_df['DlyCalDt'] == end_date][['Ticker', 'DlyCap']]
    last_day_caps = last_day_caps[last_day_caps['Ticker'].isin(Y_full.columns)]
    last_day_caps = last_day_caps.dropna()
    top_p_stocks = last_day_caps.nlargest(p_stocks, 'DlyCap')['Ticker'].tolist()
    
    Y = Y_full[top_p_stocks].values.astype(float)
    
    # Convert to percentage if in decimal
    if np.mean(np.abs(Y)) < 0.1:
        Y = Y * 100
    
    print(f"Final shape (before transpose): {Y.shape}")
    return Y.T  # Return p x n matrix

def analyze_spectrum(returns):
    """Analyze eigenvalue spectrum with proper scaling"""
    p, n = returns.shape
    print(f"\nAnalyzing {p} stocks over {n} days")
    
    # Center the data
    returns_centered = returns - np.mean(returns, axis=1, keepdims=True)
    
    # Scale by sqrt(n) for proper normalization
    returns_scaled = returns_centered / np.sqrt(n)
    
    # Use SVD for better numerical stability
    try:
        U, s, _ = np.linalg.svd(returns_scaled, full_matrices=False)
        eigenvalues = s**2
        eigenvectors = U
    except np.linalg.LinAlgError:
        print("SVD failed, falling back to eigendecomposition")
        cov_matrix = returns_scaled @ returns_scaled.T
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.flip(eigenvalues)
        eigenvectors = np.fliplr(eigenvectors)
    
    # Compute MP bounds
    q = n/p
    sigma = np.sqrt(np.mean(np.var(returns_centered, axis=1)))
    lambda_plus = sigma**2 * (1 + 1/np.sqrt(q))**2
    lambda_minus = sigma**2 * (1 - 1/np.sqrt(q))**2
    
    # Ensure positive orientation of eigenvectors
    for i in range(eigenvectors.shape[1]):
        if np.mean(eigenvectors[:, i]) < 0:
            eigenvectors[:, i] *= -1
    
    return eigenvalues, eigenvectors, lambda_plus, lambda_minus

def plot_spectrum_and_bounds(eigenvalues, lambda_plus, lambda_minus):
    """Plot eigenvalue spectrum with MP bounds"""
    plt.figure(figsize=(12, 8))
    
    # Plot non-zero eigenvalues
    valid_mask = eigenvalues > 1e-10
    valid_eigenvalues = eigenvalues[valid_mask]
    x_range = np.arange(1, len(valid_eigenvalues) + 1)
    
    plt.plot(x_range, valid_eigenvalues, 'bo-', 
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
    plt.show()

def plot_eigenvector(eigenvector, index):
    """Plot distribution of eigenvector entries"""
    plt.figure(figsize=(12, 8))
    sns.histplot(eigenvector, bins=50, kde=True)
    
    stats_text = f'Mean: {np.mean(eigenvector):.3f}\n'
    stats_text += f'Std: {np.std(eigenvector):.3f}\n'
    stats_text += f'Skew: {stats.skew(eigenvector):.3f}\n'
    stats_text += f'Kurt: {stats.kurtosis(eigenvector):.3f}'
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel(f'Entries of Eigenvector {index}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Eigenvector {index} Entries')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Parameters
    end_date = '2021-12-31'
    lookback_days = 126
    initial_p = 2000
    output_dir = 'analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    conn = sqlite3.connect('market_database.db')
    
    try:
        # Get returns data
        returns = get_returns_data(end_date, lookback_days, initial_p, conn)
        
        # Analyze spectrum
        eigenvalues, eigenvectors, lambda_plus, lambda_minus = analyze_spectrum(returns)
        
        # Plot results
        plot_spectrum_and_bounds(eigenvalues, lambda_plus, lambda_minus)
        
        # Print summary statistics
        print("\nSpectrum Analysis:")
        print(f"MP upper bound: {lambda_plus:.2f}")
        print(f"MP lower bound: {lambda_minus:.2f}")
        print("\nLargest eigenvalues and MP ratios:")
        for i in range(min(5, len(eigenvalues))):
            ratio = eigenvalues[i] / lambda_plus
            print(f"λ_{i+1}: {eigenvalues[i]:.2f} ({ratio:.1f}x MP bound)")
        
        # Plot top eigenvectors
        for i in range(4):
            plot_eigenvector(eigenvectors[:, i], i+1)
        
        # Save results
        results = pd.DataFrame({
            'eigenvalue': eigenvalues,
            'mp_upper': lambda_plus,
            'mp_lower': lambda_minus
        })
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results.to_csv(os.path.join(output_dir, f'eigenvalue_analysis_{timestamp}.csv'))
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()