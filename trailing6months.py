import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def get_returns_data(end_date, lookback_days, p_stocks, conn):
    """
    Gets returns matrix Y (n x p) for top p stocks by market cap
    n = #obs 
    p = #stocks
    """
    print(f"\Getting data for p={p_stocks}")
    
    # first trading date
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
    
    # full data
    days_str = "','".join(trading_days)
    returns_query = f"""
    SELECT DISTINCT DlyCalDt, Ticker, DlyRet, DlyCap
    FROM russell3000
    WHERE DlyCalDt IN ('{days_str}')
    ORDER BY DlyCalDt, Ticker
    """
    
    returns_df = pd.read_sql_query(returns_query, conn)
    print(f"Initial data shape: {returns_df.shape}")
    
    # id duplicates
    duplicates = returns_df.duplicated(subset=['DlyCalDt', 'Ticker'], keep=False)
    if duplicates.any():
        print(f"\nFound {duplicates.sum()} duplicate entries")
        duplicate_records = returns_df[duplicates].sort_values(['DlyCalDt', 'Ticker'])
        print("\nDuplicate entries summary:")
        duplicate_summary = duplicate_records.groupby('Ticker').size().sort_values(ascending=False)
        print(duplicate_summary.head(10)) 
        
        output_dir = os.path.join(os.getcwd(), 'analysis_output')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        duplicate_records.to_csv(os.path.join(output_dir, f'duplicates_{timestamp}.csv'))
        print(f"\nFull duplicate details saved to duplicates_{timestamp}.csv")
        
        # keep first occurrence of each (date, ticker) <- will check if this is acc a good idea...
        returns_df = returns_df.drop_duplicates(subset=['DlyCalDt', 'Ticker'], keep='first')
        print(f"Data shape after removing duplicates: {returns_df.shape}")
    
    # returns matrix for all stocks
    Y_full = returns_df.pivot(index='DlyCalDt', columns='Ticker', values='DlyRet')
    print(f"Pivoted data shape before dropping NA: {Y_full.shape}")
    
    # prune missing data
    Y_full = Y_full.dropna(axis=1)
    print(f"Data shape after dropping NA: {Y_full.shape}")
    
    # marketcap based on last date to rank
    last_day_caps = returns_df[returns_df['DlyCalDt'] == end_date][['Ticker', 'DlyCap']]
    last_day_caps = last_day_caps[last_day_caps['Ticker'].isin(Y_full.columns)]
    print(f"Number of stocks with market cap data: {len(last_day_caps)}")
    
    top_p_stocks = last_day_caps.nlargest(p_stocks, 'DlyCap')['Ticker'].tolist()
    print(f"Selected top {len(top_p_stocks)} stocks")
    
    Y = Y_full[top_p_stocks].values
    
    # demean
    Y = Y - np.mean(Y, axis=0)
    
    n, p = Y.shape
    print(f"Final returns matrix shape: n={n}, p={p}")
    
    return Y, n, p

def analyze_eigenvalues(Y, n, p):

    print(f"\nMatrix shape: ({n}, {p})")
    
    if n < p:
        print("use dual form (Y^T Y / n)")
        C = np.dot(Y.T, Y) / n
        eigenvalues = np.linalg.eigvals(C)
    else:
        print("use primal form (YY^T / n)")
        C = np.dot(Y, Y.T) / n
        eigenvalues = np.linalg.eigvals(C)
    
    eigenvalues = np.real(eigenvalues)
    print(f"Computed {len(eigenvalues)} eigenvalues")
    print(f"Leading eigenvalue: {np.max(eigenvalues):.4f}")
    
    return eigenvalues

def plot_eigenvalue_histogram(eigenvalues, n, p, output_dir, timestamp):
    """Plot histogram of eigenvalues"""
    plt.figure(figsize=(10, 6))
    plt.hist(eigenvalues, bins=50, edgecolor='black')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Eigenvalues\n(n={n}, p={p})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'eigenvalue_hist_p{p}_{timestamp}.png'))
    plt.close()

def analyze_2021_eigenvalues():
    """Analyze eigenvalues for different p values using 2021 end date"""
    end_date = '2021-12-31'
    lookback_days = 126
    P_values = [50, 100, 200, 500, 1000]
    results = []
    
    print(f"Starting analysis for end date {end_date} with {lookback_days} lookback days")
    print(f"Will analyze the following p values: {P_values}")
    
    output_dir = os.path.join(os.getcwd(), 'analysis_output')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    conn = sqlite3.connect('market_database.db')
    
    try:
        for p in P_values:
            print(f"\n{'='*50}")
            print(f"Processing p={p}")
            print(f"{'='*50}")
            
            #  returns matrix
            Y, n, p_actual = get_returns_data(end_date, lookback_days, p, conn)
            
            #  eigenvalues
            eigenvalues = analyze_eigenvalues(Y, n, p_actual)
            
            plot_eigenvalue_histogram(eigenvalues, n, p_actual, output_dir, timestamp)
            
            result = {
                'p': p_actual,
                'n': n,
                'leading_eigenvalue': np.max(eigenvalues),
                'mean_eigenvalue': np.mean(eigenvalues),
                'median_eigenvalue': np.median(eigenvalues)
            }
            results.append(result)
            print(f"\nResults for p={p}:")
            print(result)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        conn.close()
    
    if not results:
        print("No results were generated!")
        return None
    
    results_df = pd.DataFrame(results)
    print("\nFinal results dataframe:")
    print(results_df)
    
    #  leading eigenvalue vs p
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['p'], results_df['leading_eigenvalue'], 'o-')
    plt.xlabel('Number of Stocks (p)')
    plt.ylabel('Leading Eigenvalue')
    plt.title(f'Leading Eigenvalue vs p\n(n={lookback_days} days, end={end_date})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'leading_eigenvalue_{timestamp}.png'))
    plt.close()
    
    results_df.to_csv(os.path.join(output_dir, f'eigenvalue_results_{timestamp}.csv'))
    
    return results_df

if __name__ == "__main__":
    results = analyze_2021_eigenvalues()
    if results is not None:
        print("\nResults:")
        print(results)
    else:
        print("error")