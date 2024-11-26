import sqlite3
from datetime import datetime, timedelta
import numpy as np

def get_return_matrix(end_date, n_days, min_returns_pct=100):
    """
    Generates p x n return matrix given a specific date range (queries SQLite)
    
    Parameters:
    -----------
    end_date        : str   [The end date in format 'YYYY-MM-DD']
    n_days          : int   [Number of days to look back]
    min_returns_pct : float [Minimum percentage of returns required (default: 100%)]
    
    Returns:
    --------
    returns         : dict [Dictionary with tickers as keys and lists of returns as values]
    dates           : list [List of dates corresponding to the returns]
    """
    conn = sqlite3.connect('stock_database.db')
    cursor = conn.cursor()
    
    # obtain n_days before end_date
    cursor.execute('''
    SELECT DISTINCT DlyCalDt 
    FROM stock_data 
    WHERE DlyCalDt <= ?
    ORDER BY DlyCalDt DESC
    LIMIT ?
    ''', (end_date, n_days))
    
    dates = [date[0] for date in cursor.fetchall()][::-1]  # change to chronological order
    
    if len(dates) < n_days:
        raise ValueError(f"Not enough dates available before {end_date}")
    
    # find tickers with sufficient data in the period
    min_dates = int(n_days * min_returns_pct / 100)
    cursor.execute('''
    SELECT Ticker, COUNT(DISTINCT DlyCalDt) as date_count
    FROM stock_data
    WHERE DlyCalDt IN ({})
    GROUP BY Ticker
    HAVING date_count >= ?
    '''.format(','.join('?' * len(dates))), (*dates, min_dates))
    
    valid_tickers = [row[0] for row in cursor.fetchall()]
    
    returns = {}
    for ticker in valid_tickers:
        cursor.execute('''
        SELECT DlyCalDt, DlyRet
        FROM stock_data
        WHERE Ticker = ?
        AND DlyCalDt IN ({})
        ORDER BY DlyCalDt
        '''.format(','.join('?' * len(dates))), (ticker, *dates))
        
        ticker_returns = dict(cursor.fetchall())
        
        # [not called by default] using 0 for missing dates
        returns[ticker] = [ticker_returns.get(date, 0) for date in dates]
    
    conn.close()
    return returns, dates

def demonstrate_matrix_usage():
    """
    sample matrix function, last 5 days
    """
    conn = sqlite3.connect('stock_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(DlyCalDt) FROM stock_data")
    max_date = cursor.fetchone()[0]
    conn.close()
    
    returns, dates = get_return_matrix(max_date, 5)
    
    print(f"Return matrix generated for {len(returns)} stocks over {len(dates)} days")
    print("\nDates included:")
    print(dates)
    
    # Print first few stocks as example
    print("\nExample returns for first 3 stocks:")
    for i, (ticker, rets) in enumerate(returns.items()):
        if i < 3:
            print(f"{ticker}: {rets}")
    
    all_returns = np.array(list(returns.values()))
    print("\nBasic statistics:")
    print(f"Mean return: {np.mean(all_returns):.6f}")
    print(f"Std deviation: {np.std(all_returns):.6f}")
    print(f"Min return: {np.min(all_returns):.6f}")
    print(f"Max return: {np.max(all_returns):.6f}")
    
    return returns, dates

# Get returns matrix
returns, dates = get_return_matrix('2003-01-24', 10)

returns, dates = demonstrate_matrix_usage()


