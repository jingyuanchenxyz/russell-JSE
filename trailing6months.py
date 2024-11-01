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
    print(f"\n got data for p={p_stocks}")
    
    # get trading days: DlyCalDt means daily calander date
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
    
    # get full data: Calender Date, Name (Ticker), Daily Return and Daily Market Cap
    days_str = "','".join(trading_days)
    returns_query = f"""
    SELECT DISTINCT DlyCalDt, Ticker, DlyRet, DlyCap
    FROM russell3000
    WHERE DlyCalDt IN ('{days_str}')
    ORDER BY DlyCalDt, Ticker
    """
    
    returns_df = pd.read_sql_query(returns_query, conn)
    
    # Convert DlyRet and DlyCap to float (from str)
    returns_df['DlyRet'] = pd.to_numeric(returns_df['DlyRet'], errors='coerce')
    returns_df['DlyCap'] = pd.to_numeric(returns_df['DlyCap'], errors='coerce')
    
    # Remove duplicates
    returns_df = returns_df.drop_duplicates(subset=['DlyCalDt', 'Ticker'], keep='first')
    
    # returns matrix
    Y_full = returns_df.pivot(index='DlyCalDt', columns='Ticker', values='DlyRet')
    Y_full = Y_full.dropna(axis=1)
    
    # top p stocks by market cap
    last_day_caps = returns_df[returns_df['DlyCalDt'] == end_date][['Ticker', 'DlyCap']]
    last_day_caps = last_day_caps[last_day_caps['Ticker'].isin(Y_full.columns)]
    last_day_caps = last_day_caps.dropna()  # remove any NaN market caps
    top_p_stocks = last_day_caps.nlargest(p_stocks, 'DlyCap')['Ticker'].tolist()
    
    Y = Y_full[top_p_stocks].values.astype(float)  # assert float type
    # Y = Y - np.mean(Y, axis=0)  # demean
    
    print(f"final shape (before transpose): {Y.shape}")
    return Y.T  # p x n

def calculate_max_eigenvalue(Y):
    """ largest eigenvalue of Y^T Y / n"""
    n = Y.shape[1]
    C = np.dot(Y, Y.T) / n
    eigenvalues = np.linalg.eigvals(C)
    return np.max(np.real(eigenvalues))

def analyze_decreasing_subsets(Y, num_runs=20):
    """
    iter: check eigenvalues for decreasing subsets of stocks
    returns: dict with p_values and eigenvalue statistics
    """
    p, n = Y.shape
    results = {
        'p_values': [],
        'mean_eigenvalues': [],
        'std_eigenvalues': [],
        'min_eigenvalues': [],
        'max_eigenvalues': []
    }
    
    for k in range(0, (p-100)//100 + 1):
        subset_size = p - k*100
        if subset_size < 100:  # end if subset gets too small
            break
            
        print(f"\nAnalyzing subset size p={subset_size}")
        eigenvalues = []
        
        for run in range(num_runs):
            # randomly sample subset_size stocks
            stock_indices = np.random.choice(p, subset_size, replace=False)
            Y_subset = Y[stock_indices, :]
            
            # largest eigenvalue
            max_eigenvalue = calculate_max_eigenvalue(Y_subset)
            eigenvalues.append(max_eigenvalue)
        
        results['p_values'].append(subset_size)
        results['mean_eigenvalues'].append(np.mean(eigenvalues))
        results['std_eigenvalues'].append(np.std(eigenvalues))
        results['min_eigenvalues'].append(np.min(eigenvalues))
        results['max_eigenvalues'].append(np.max(eigenvalues))
        
    return results

def plot_results(results, output_dir):
    plt.figure(figsize=(16, 10), dpi=400)
    
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'serif', 'serif': ['Times']})
    
    deluge = "#7C71AD"
    yellow = '#FFAC00'
    deluge_a = plt.matplotlib.colors.colorConverter.to_rgba(deluge, alpha=0.50)
    deluge_b = plt.matplotlib.colors.colorConverter.to_rgba(deluge, alpha=0.25)
    
    data_to_plot = []
    p_values = list(reversed(results['p_values']))
    
    for p_idx, p_value in enumerate(p_values):
        orig_idx = results['p_values'].index(p_value)
        mean = results['mean_eigenvalues'][orig_idx]
        std = results['std_eigenvalues'][orig_idx]
        points = np.random.normal(mean, std, 100) 
        data_to_plot.append(points)
    
    bp = plt.boxplot(data_to_plot, 
                    notch=True,
                    widths=0.44 * np.ones(len(p_values)),
                    whis=[1, 99],
                    patch_artist=True)
    
    for i in range(len(p_values)):
        # Boxes
        plt.setp(bp['boxes'][i], linewidth=4,
                facecolor=deluge_a, edgecolor=deluge_b)
        
        # Outlier points
        plt.setp(bp['fliers'][i], marker='o',
                alpha=0.6, markersize=7.5,
                markerfacecolor=yellow,
                markeredgecolor=deluge)
        
        # Median lines
        plt.setp(bp['medians'][i], color=deluge,
                linewidth=4, alpha=0.5)
        
        # Whiskers
        plt.setp(bp['whiskers'][2*i], color=yellow,
                linewidth=3, alpha=0.8)
        plt.setp(bp['whiskers'][2*i+1], color=yellow,
                linewidth=3, alpha=0.8)
        
        # Caps
        plt.setp(bp['caps'][2*i], color=deluge,
                linewidth=4, alpha=0.5)
        plt.setp(bp['caps'][2*i+1], color=deluge,
                linewidth=4, alpha=0.5)
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.xlabel('Number of Stocks (p)', fontsize=24)
    plt.ylabel('Largest Eigenvalue', fontsize=24)
    plt.xticks(range(1, len(p_values) + 1),
               [str(int(p)) for p in p_values],
               fontsize=20, rotation=0)
    plt.yticks(fontsize=20)
    
    title = f"n={min(p_values)}, runs={len(data_to_plot[0])}\n"
    plt.title(title, fontsize=20, pad=20)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(os.path.join(output_dir, f'eigenvalue_analysis_{timestamp}.png'),
                transparent=True,
                dpi=300,  
                bbox_inches='tight', 
                pad_inches=0.1)  
    plt.close()
    
def main():
    end_date = '2021-12-31'
    lookback_days = 126
    initial_p = 2000  # starting p
    num_runs = 20
    
    output_dir = os.path.join(os.getcwd(), 'analysis_output')
    os.makedirs(output_dir, exist_ok=True)
    
    conn = sqlite3.connect('market_database.db')
    
    try:
        Y = get_returns_data(end_date, lookback_days, initial_p, conn)
        print(f"\nInitial data matrix shape: {Y.shape}")
        
        results = analyze_decreasing_subsets(Y, num_runs)
        
        plot_results(results, output_dir)
        
        results_df = pd.DataFrame({
            'p': results['p_values'],
            'mean_eigenvalue': results['mean_eigenvalues'],
            'std_eigenvalue': results['std_eigenvalues'],
            'min_eigenvalue': results['min_eigenvalues'],
            'max_eigenvalue': results['max_eigenvalues']
        })
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(os.path.join(output_dir, f'eigenvalue_results_{timestamp}.csv'))
        print("\nResults summary:")
        print(results_df)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        conn.close()

if __name__ == "__main__":
    main()