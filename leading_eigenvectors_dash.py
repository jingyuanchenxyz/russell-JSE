import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import sqlite3
import time
from scipy import stats


# Create indices if they don't exist
@st.cache_resource  # This ensures we only try to create indices once per session
def create_indices(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dlycaldt ON russell3000(DlyCalDt)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dlycaldt_ticker ON russell3000(DlyCalDt, Ticker)")
            print("Database indices checked/created successfully")
    except Exception as e:
        print(f"Warning: Could not create indices: {e}")

plt.ioff()

class MarketCapAnalyzer:
    def __init__(self, database_path='market_database.db'):
        self.db_path = database_path
        # Create indices when initializing
        create_indices(self.db_path)
        self.dates = None
        self._load_dates()
    
    @st.cache_data  # Cache the dates query
    def _load_dates(self):
        """Load available dates from database with caching"""
        with sqlite3.connect(self.db_path) as conn:
            self.dates = pd.read_sql_query(
                "SELECT DISTINCT DlyCalDt FROM russell3000 ORDER BY DlyCalDt",
                conn
            )['DlyCalDt'].tolist()
    
    @staticmethod
    @lru_cache(maxsize=32)  # Cache eigenvector stats calculations
    def compute_eigenvector_stats_cached(eigenvector_tuple):
        """Cached version of statistical computations"""
        evec = np.array(eigenvector_tuple)
        return {
            'Mean': np.mean(evec),
            'Std': np.std(evec),
            'Skewness': stats.skew(evec),
            'Kurtosis': stats.kurtosis(evec),
            'Min': np.min(evec),
            'Max': np.max(evec),
            'Median': np.median(evec)
        }
            
    def get_window_data(self, end_date, lookback_days):
        """Get returns matrix for market cap ranked stocks"""
        with sqlite3.connect(self.db_path) as conn:
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
            sorted_stocks = last_day_caps.sort_values('DlyCap', ascending=False)
            
            Y = Y_full[sorted_stocks['Ticker']].values.astype(float)
            Y = Y - np.mean(Y, axis=0)
            
            return Y.T, sorted_stocks, Y_full.columns
            
    def analyze_market_cap_ranked(self, Y, market_caps, num_stocks):
        """Analyze stocks ranked by market cap"""
        Y_subset = Y[:num_stocks, :]
        eigenvalues, eigenvectors = self.compute_eigen(Y_subset)
        # Remove normalization
        return eigenvalues, eigenvectors

    
    @staticmethod
    def compute_eigen(Y, top_k=4):
        """Compute eigendecomposition using SVD"""
        n = Y.shape[1]
        U, S, Vh = np.linalg.svd(Y)
        eigenvalues = (S**2) / n
        return eigenvalues[:top_k], U[:, :top_k]

def main():
    st.title("Market Cap Ranked Eigenvector Analysis")
    
    analyzer = MarketCapAnalyzer()
    
    try:
        min_date = datetime.strptime(min(analyzer.dates), '%Y-%m-%d').date()
        max_date = datetime.strptime(max(analyzer.dates), '%Y-%m-%d').date()
        available_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in analyzer.dates]
        available_dates.sort()
        
        if not available_dates:
            st.error("No dates available in the database.")
            return
        
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
                
                tab1, tab2 = st.tabs(["Analysis", "Details"])
                
                with tab1:
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.subheader("Eigenvalue Spectrum")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        _, S, _ = np.linalg.svd(ret_matrix[:num_stocks])
                        all_eigenvalues = (S**2) / ret_matrix.shape[1]
                        
                        sns.histplot(
                            all_eigenvalues,
                            bins=500,
                            ax=ax,
                            stat='density',
                            kde=True,
                            color='blue',
                            alpha=0.6
                        )
                        
                        for i, ev in enumerate(eigenvalues):
                            ax.axvline(
                                ev, 
                                color=f'C{i}',
                                linestyle='--',
                                label=f'λ{i+1}: {ev:.3f}'
                            )
                        
                        ax.set_xlabel('Eigenvalues')
                        ax.set_ylabel('Density')
                        ax.set_title('Eigenvalue Distribution')
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)
                        plt.close()
                    
                    with col4:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                        axes = axes.ravel()
                        
                        for i in range(4):
                            sns.histplot(
                                eigenvectors[:, i],
                                bins=50,
                                ax=axes[i],
                                kde=True
                            )
                            axes[i].set_title(f'Eigenvector {i+1}')
                            axes[i].set_xlabel('Values')  
                            axes[i].set_ylabel('Frequency')
                            axes[i].grid(True)
                            
                            moments_text = f'μ={np.mean(eigenvectors[:, i]):.3f}\nσ={np.std(eigenvectors[:, i]):.3f}\n'
                            moments_text += f'Skew={stats.skew(eigenvectors[:, i]):.3f}\nKurt={stats.kurtosis(eigenvectors[:, i]):.3f}'
                            axes[i].text(
                                0.95, 0.95,
                                moments_text,
                                transform=axes[i].transAxes,
                                verticalalignment='top',
                                horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                            )
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                
                with tab2:
                    st.subheader(f"Top {num_stocks} Stocks by Market Cap")
                    top_stocks_df = market_caps.head(num_stocks)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.histplot(data=top_stocks_df, x='DlyCap', bins=50, ax=ax)
                    ax.set_xlabel('Market Cap')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    plt.close()
                    
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
                    
                    if st.checkbox("Show Stock Details"):
                        st.dataframe(top_stocks_df[['Ticker', 'DlyCap']].reset_index(drop=True))
            
            else:
                st.error("No data available for selected period")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check the database connection and data format.")

if __name__ == "__main__":
    main()