import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import sqlite3
import time

class MarketCapAnalyzer:
    def __init__(self, database_path='market_database.db'):
        self.db_path = database_path
        self.dates = None
        self._load_dates()
    
    def _load_dates(self):
        """Load available dates from database"""
        with sqlite3.connect(self.db_path) as conn:
            self.dates = pd.read_sql_query(
                "SELECT DISTINCT DlyCalDt FROM russell3000 ORDER BY DlyCalDt",
                conn
            )['DlyCalDt'].tolist()
            
    def get_window_data(self, end_date, lookback_days):
        """Get returns matrix for market cap ranked stocks"""
        with sqlite3.connect(self.db_path) as conn:
            # Get trading days
            days_query = """
            SELECT DISTINCT DlyCalDt
            FROM russell3000
            WHERE DlyCalDt <= ?
            ORDER BY DlyCalDt DESC
            LIMIT ?
            """
            trading_days = pd.read_sql_query(
                days_query,
                conn,
                params=(end_date, lookback_days)
            )
            trading_days = sorted(trading_days['DlyCalDt'].tolist())
            
            if len(trading_days) < lookback_days:
                return None, None, None
            
            # Get full data
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
            
            # Get and sort stocks by market cap
            last_day_caps = returns_df[returns_df['DlyCalDt'] == end_date][['Ticker', 'DlyCap']]
            last_day_caps = last_day_caps[last_day_caps['Ticker'].isin(Y_full.columns)]
            last_day_caps = last_day_caps.dropna()
            sorted_stocks = last_day_caps.sort_values('DlyCap', ascending=False)
            
            # Final returns matrix
            Y = Y_full[sorted_stocks['Ticker']].values.astype(float)
            Y = Y - np.mean(Y, axis=0)  # demean
            
            return Y.T, sorted_stocks, Y_full.columns
            
    def analyze_market_cap_ranked(self, Y, market_caps, num_stocks):
        """Analyze stocks ranked by market cap"""
        Y_subset = Y[:num_stocks, :]
        eigenvalues, eigenvectors = self.compute_eigen(Y_subset)
        # Normalize eigenvectors
        normalized_eigenvectors = np.array([
            evec / np.mean(evec) for evec in eigenvectors.T
        ]).T
        return eigenvalues, normalized_eigenvectors
    
    @staticmethod
    def compute_eigen(Y, top_k=4):
        """Compute eigendecomposition using SVD for positive eigenvalues"""
        n = Y.shape[1]
        U, S, Vh = np.linalg.svd(Y)
        # Eigenvalues are squares of singular values divided by n
        eigenvalues = (S**2) / n
        # Return top k eigenvalues and corresponding eigenvectors
        return eigenvalues[:top_k], U[:, :top_k]

def main():
    st.title("Market Cap Ranked Eigenvector Analysis")
    
    # Initialize analyzer
    analyzer = MarketCapAnalyzer()
    
    try:
        # Convert dates
        min_date = datetime.strptime(min(analyzer.dates), '%Y-%m-%d').date()
        max_date = datetime.strptime(max(analyzer.dates), '%Y-%m-%d').date()
        available_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in analyzer.dates]
        available_dates.sort()
        
        if not available_dates:
            st.error("No dates available in the database.")
            return
        
        # Date selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_date = st.date_input(
                "Select Analysis Date",
                value=available_dates[-1],
                min_value=available_dates[0],
                max_value=available_dates[-1]
            )
            
            if selected_date not in available_dates:
                st.warning(f"Selected date {selected_date} is not a trading day.")
                nearest_date = min(available_dates, key=lambda x: abs(x - selected_date))
                st.info(f"Nearest available trading day: {nearest_date}")
                selected_date = nearest_date
        
        with col2:
            lookback = st.number_input(
                "Lookback Period (trading days)",
                min_value=30,
                max_value=252,
                value=126,
                step=1
            )
        
        num_stocks = st.slider(
            "Number of Stocks by Market Cap",
            min_value=100,
            max_value=2000,
            value=100,
            step=100,
            help="Select stocks starting from largest market cap"
        )
        
        # Process data
        start_time = time.time()
        
        with st.spinner('Processing data...'):
            ret_matrix, market_caps, all_stocks = analyzer.get_window_data(
                selected_date.strftime('%Y-%m-%d'),
                lookback
            )
            
            if ret_matrix is not None:
                eigenvalues, eigenvectors = analyzer.analyze_market_cap_ranked(
                    ret_matrix,
                    market_caps,
                    num_stocks
                )
                
                # Create visualization tabs
                tab1, tab2 = st.tabs(["Analysis", "Details"])
                
                with tab1:
                    # Create two columns for eigenvalue plots
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Full spectrum histogram
                        st.subheader("Eigenvalue Spectrum")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        _, S, _ = np.linalg.svd(ret_matrix[:num_stocks])
                        all_eigenvalues = (S**2) / ret_matrix.shape[1]
                        sns.histplot(all_eigenvalues, bins=30, ax=ax)
                        ax.set_xlabel('Eigenvalues')
                        ax.set_ylabel('Frequency')
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close()
                    
                    with col4:
                        st.subheader("Top 4 Eigenvalues")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(len(eigenvalues)), eigenvalues)
                        ax.set_xticks(range(len(eigenvalues)))
                        ax.set_xticklabels([f'λ{i+1}' for i in range(len(eigenvalues))])
                        ax.set_ylabel('Eigenvalue')
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close()
                    
                    # Add subheader for eigenvectors
                    st.subheader("Top 4 Eigenvectors")
                    
                    # Create 2x2 grid for eigenvector histograms
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    axes = axes.ravel()
                    
                    for i in range(4):
                        sns.histplot(
                            eigenvectors[:, i], 
                            bins=20, 
                            ax=axes[i]
                        )
                        axes[i].set_title(f'Eigenvector {i+1}')
                        axes[i].set_xlabel('Normalized Entries')
                        axes[i].set_ylabel('Frequency')
                        axes[i].grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with tab2:
                    # Add box plot of eigenvectors
                    st.subheader("Eigenvector Distributions")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    data_to_plot = [eigenvectors[:, i] for i in range(4)]
                    bp = ax.boxplot(
                        data_to_plot, 
                        labels=[f'Eigenvector {i+1}' for i in range(4)],
                        showmeans=True
                    )
                    ax.set_xlabel('Eigenvectors')
                    ax.set_ylabel('Normalized Entries')
                    ax.grid(True)
                    ax.set_ylim(-100, 100)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show market cap distribution
                    st.subheader(f"Top {num_stocks} Stocks by Market Cap")
                    top_stocks_df = market_caps.head(num_stocks)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.histplot(data=top_stocks_df, x='DlyCap', bins=30, ax=ax)
                    ax.set_xlabel('Market Cap')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show performance metrics
                    col5, col6 = st.columns(2)
                    with col5:
                        st.metric(
                            "Matrix Size",
                            f"{ret_matrix[:num_stocks].shape[0]}×{ret_matrix[:num_stocks].shape[1]}"
                        )
                        st.metric(
                            "Total Variance Explained",
                            f"{(sum(eigenvalues)/np.trace(ret_matrix[:num_stocks] @ ret_matrix[:num_stocks].T/ret_matrix[:num_stocks].shape[1])*100):.2f}%"
                        )
                    
                    with col6:
                        process_time = time.time() - start_time
                        st.metric("Processing Time", f"{process_time:.3f}s")
                        st.metric(
                            "Average Market Cap",
                            f"${top_stocks_df['DlyCap'].mean():,.2f}M"
                        )
                    
                    # Show top stocks table
                    if st.checkbox("Show Stock Details"):
                        st.dataframe(top_stocks_df[['Ticker', 'DlyCap']].reset_index(drop=True))
            
            else:
                st.error("No data available for selected period")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check the database connection and data format.")

if __name__ == "__main__":
    main()