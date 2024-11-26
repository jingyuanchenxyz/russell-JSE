import sqlite3
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import traceback

def get_returns_data(end_date, lookback_days, p_stocks, conn):
    """
    Gets returns matrix Y (p x n) for top p stocks by market cap
    """
    print(f"\n got data for p={p_stocks}")
    
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
        raise ValueError(f"only found {len(trading_days)} trading days")
    
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
    
    Y_full = returns_df.pivot(index='DlyCalDt', columns='Ticker', values='DlyRet')
    Y_full = Y_full.dropna(axis=1)
    
    last_day_caps = returns_df[returns_df['DlyCalDt'] == end_date][['Ticker', 'DlyCap']]
    last_day_caps = last_day_caps[last_day_caps['Ticker'].isin(Y_full.columns)]
    last_day_caps = last_day_caps.dropna()
    top_p_stocks = last_day_caps.nlargest(p_stocks, 'DlyCap')['Ticker'].tolist()
    
    Y = Y_full[top_p_stocks].values.astype(float)
    print(f"final shape (before transpose): {Y.shape}")
    return Y.T

def estimate_factor_model(Y):
    """
    Estimate factor model using PCA with proper scaling
    Y should be p x n (stocks x time)
    Returns beta (normalized to mean 1), factor returns, residuals, and eigenvalue
    """
    p, n = Y.shape
    
    # Convert returns to percentages if they're not already (assuming decimals)
    if np.mean(np.abs(Y)) < 0.1:  # If returns are in decimal form
        Y = Y * 100
    
    # Demean each stock's returns
    Y_demean = Y - np.mean(Y, axis=1, keepdims=True)
    
    # Use PCA
    pca = PCA(n_components=1)
    pca.fit(Y_demean.T)
    
    # Extract components
    leading_eigval = pca.explained_variance_[0]
    leading_eigvec = pca.components_[0]
    f = pca.fit_transform(Y_demean.T)[:, 0]
    
    # Ensure positive mean direction
    if np.mean(leading_eigvec) < 0:
        leading_eigvec = -leading_eigvec
        f = -f
    
    # Normalize beta to mean 1
    beta = leading_eigvec / np.mean(leading_eigvec)
    
    # Compute residuals
    epsilon = Y_demean - np.outer(beta, f)
    
    # Calculate variances (no need for extra scaling)
    sigma_f_sq = np.var(f)
    delta_sq = np.mean(np.var(epsilon, axis=1))
    
    return beta, f, epsilon, leading_eigval, sigma_f_sq, delta_sq

def analyze_theoretical_relationships(Y, annualize=True):
    """
    Analyze factor model theoretical relationships
    """
    p, n = Y.shape
    beta, f, epsilon, leading_eigval, sigma_f_sq, delta_sq = estimate_factor_model(Y)
    
    # Calculate quantities
    beta_var = np.var(beta)  # τ²
    
    # Annualization
    scale = np.sqrt(252) if annualize else 1  # Use sqrt(252) for volatility
    
    # Theoretical vs Observed slopes
    observed_slope = leading_eigval / p
    theoretical_slope = (1 + beta_var)  # Should match PCA eigenvalue/p
    
    results = {
        'p': p,
        'n': n,
        'beta_mean': np.mean(beta),
        'beta_variance': beta_var,
        'factor_variance': sigma_f_sq,
        'observed_slope': observed_slope,
        'theoretical_slope': theoretical_slope,
        'factor_vol_annual': np.sqrt(sigma_f_sq) * scale,  # Annualized volatility in %
        'residual_vol_annual': np.sqrt(delta_sq) * scale   # Annualized volatility in %
    }
    
    return results

def plot_factor_analysis(results_list, output_dir):
    """Plot analysis results"""
    plt.figure(figsize=(16, 10), dpi=300, facecolor='white')
    
    plt.rc('text', usetex=False)
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    
    p_values = [r['p'] for r in results_list]
    observed_slopes = [r['observed_slope'] for r in results_list]
    theoretical_slopes = [r['theoretical_slope'] for r in results_list]
    
    print("\nPlotting Values:")
    for p, obs, theo in zip(p_values, observed_slopes, theoretical_slopes):
        print(f"p={p}, observed={obs:.4f}, theoretical={theo:.4f}")
    
    plt.plot(p_values, observed_slopes, 'bo-', label='Observed (λ₁/p)', alpha=0.7)
    plt.plot(p_values, theoretical_slopes, 'r--', label='Theoretical (1 + τ²)', alpha=0.7)
    
    plt.xlabel('Number of Stocks (p)', fontsize=24)
    plt.ylabel('Slope', fontsize=24)
    plt.title('Comparison of Observed vs Theoretical Slopes', fontsize=20, pad=20)
    plt.legend(fontsize=16)
    
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(True, alpha=0.3)
    
    plt.gca().invert_xaxis()
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(output_dir, f'factor_analysis_{timestamp}.png'),
                facecolor='white',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

def main():
    """Main execution function"""
    end_date = '2021-12-31'
    lookback_days = 126
    initial_p = 2000
    
    output_dir = os.path.join(os.getcwd(), 'analysis_output')
    os.makedirs(output_dir, exist_ok=True)
    
    conn = sqlite3.connect('market_database.db')
    
    try:
        results_list = []
        Y_full = get_returns_data(end_date, lookback_days, initial_p, conn)
        
        for k in range(0, (initial_p-100)//100 + 1):
            p = initial_p - k*100
            if p < 100:
                break
            
            print(f"\nAnalyzing p={p}")
            Y = Y_full[:p, :]
            
            results = analyze_theoretical_relationships(Y, annualize=True)
            results_list.append(results)
            
            print("\nResults for p =", p)
            print(f"Beta mean (should be 1): {results['beta_mean']:.6f}")
            print(f"Beta variance (τ²): {results['beta_variance']:.6f}")
            print(f"Observed slope (λ₁/p): {results['observed_slope']:.6f}")
            print(f"Theoretical slope (1+τ²): {results['theoretical_slope']:.6f}")
            print(f"Factor volatility (annualized): {results['factor_vol_annual']:.2f}%")
            print(f"Residual volatility (annualized): {results['residual_vol_annual']:.2f}%")
        
        plot_factor_analysis(results_list, output_dir)
        
        results_df = pd.DataFrame(results_list)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(os.path.join(output_dir, f'factor_analysis_{timestamp}.csv'))
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print(traceback.format_exc())
    finally:
        conn.close()

if __name__ == "__main__":
    main()