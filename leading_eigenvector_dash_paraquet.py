import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import time
from scipy import stats
import requests
from io import BytesIO
import matplotlib.pyplot as plt

@st.cache_data(ttl=3600, show_spinner=False)
def load_parquet_from_github():
    url = "r3000hist.parquet"
    try:
        df = pd.read_parquet(url)
        df['DlyCalDt'] = pd.to_datetime(df['DlyCalDt'])
        df['DlyRet'] = pd.to_numeric(df['DlyRet'], errors='coerce')
        df['DlyCap'] = pd.to_numeric(df['DlyCap'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

class MarketCapAnalyzer:
    def __init__(self):
        self.dates = None
        self._load_dates()
    
    def _load_dates(self):
        df = load_parquet_from_github()
        if df is not None:
            self.dates = df['DlyCalDt'].dt.strftime('%Y-%m-%d').unique().tolist()
            self.dates.sort()

    def _load_dates(self):
        """
        Loads and sorts the available dates from the Parquet file.
        """
        df = load_parquet_from_github()
        if df is not None:
            df_dates = df[['DlyCalDt']]
            # Ensure 'DlyCalDt' is datetime
            if not pd.api.types.is_datetime64_any_dtype(df_dates['DlyCalDt']):
                df_dates['DlyCalDt'] = pd.to_datetime(df_dates['DlyCalDt'], errors='coerce')
            self.dates = df_dates['DlyCalDt'].dt.strftime('%Y-%m-%d').unique().tolist()
            self.dates.sort()
    
    def compute_eigenvector_stats(self, eigenvectors):
        """
        Computes statistical moments for each eigenvector.
        """
        stats_data = []
        for i in range(eigenvectors.shape[1]):
            evec = eigenvectors[:, i]
            stats_data.append({
                'Eigenvector': f'EV{i+1}',
                'Mean': np.mean(evec),
                'Std': np.std(evec),
                'Skewness': stats.skew(evec),
                'Kurtosis': stats.kurtosis(evec),
                'Min': np.min(evec),
                'Max': np.max(evec),
                'Median': np.median(evec)
            })
        return pd.DataFrame(stats_data)
            
    def get_window_data(self, end_date, lookback_days):
        try:
            df = load_parquet_from_github()
            if df is None:
                return None, None, None
            
            end_date_dt = pd.to_datetime(end_date)
            df_filtered = df[df['DlyCalDt'] <= end_date_dt].copy()
            
            # Get last N trading days efficiently
            trading_days = df_filtered['DlyCalDt'].drop_duplicates().nlargest(lookback_days)
            if len(trading_days) < lookback_days:
                return None, None, None
            
            # Memory efficient filtering
            returns_df = df_filtered[df_filtered['DlyCalDt'].isin(trading_days)]
            returns_df = returns_df.drop_duplicates(subset=['DlyCalDt', 'Ticker'])
            
            # Efficient pivoting with error handling
            Y_full = returns_df.pivot(index='DlyCalDt', columns='Ticker', values='DlyRet')
            Y_full = Y_full.dropna(axis=1)  # Remove columns with any NaN
            
            last_day_caps = returns_df[
                returns_df['DlyCalDt'] == end_date_dt
            ][['Ticker', 'DlyCap']].dropna()
            
            last_day_caps = last_day_caps[last_day_caps['Ticker'].isin(Y_full.columns)]
            sorted_stocks = last_day_caps.sort_values('DlyCap', ascending=False)
            
            Y = Y_full[sorted_stocks['Ticker']].values.astype(np.float32)  # Use float32 for memory
            Y = Y - np.mean(Y, axis=0)
            
            return Y.T, sorted_stocks, Y_full.columns.tolist()
            
        except Exception as e:
            st.error(f"Error in get_window_data: {str(e)}")
            return None, None, None
            
    def analyze_market_cap_ranked(self, Y, market_caps, num_stocks):
        """
        Analyzes stocks ranked by market cap by computing eigenvalues and eigenvectors.
        """
        Y_subset = Y[:num_stocks, :]
        eigenvalues, eigenvectors = self.compute_eigen(Y_subset)
        
        # Only mean-zero the first eigenvector
        eigenvectors_normalized = eigenvectors.copy()
        eigenvectors_normalized[:, 0] = eigenvectors[:, 0] - np.mean(eigenvectors[:, 0])
        
        return eigenvalues, eigenvectors_normalized
    
    @staticmethod
    def compute_eigen(Y, top_k=4):
        try:
            n = Y.shape[1]
            # Use more efficient SVD computation
            U, S, Vh = np.linalg.svd(Y, full_matrices=False, compute_uv=True)
            eigenvalues = (S**2) / n
            return eigenvalues[:top_k], U[:, :top_k]
        except np.linalg.LinAlgError:
            st.error("SVD computation failed. Data might be singular.")
            return None, None
        except Exception as e:
            st.error(f"Error in compute_eigen: {str(e)}")
            return None, None

def main():
    st.title("Market Cap Ranked Eigenvector Analysis")
    
    analyzer = MarketCapAnalyzer()
    
    try:
        if not analyzer.dates:
            st.error("No dates available in the dataset.")
            return
        
        min_date = datetime.strptime(min(analyzer.dates), '%Y-%m-%d').date()
        max_date = datetime.strptime(max(analyzer.dates), '%Y-%m-%d').date()
        available_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in analyzer.dates]
        available_dates.sort()
        
        if not available_dates:
            st.error("No dates available in the dataset.")
            return
        
        # Layout: Two columns for date and lookback inputs
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
        
        # Slider for number of top stocks
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
                
                # Create two tabs: Analysis and Details
                tab1, tab2 = st.tabs(["Analysis", "Details"])
                
                with tab1:
                    # Two columns within the Analysis tab
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.subheader("Eigenvalue Spectrum")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Compute all eigenvalues for histogram
                        _, S, _ = np.linalg.svd(ret_matrix[:num_stocks], full_matrices=False)
                        all_eigenvalues = (S**2) / ret_matrix.shape[1]
                        
                        sns.histplot(
                            all_eigenvalues,
                            bins=50,
                            ax=ax,
                            stat='density',
                            kde=True,
                            color='blue',
                            alpha=0.6
                        )
                        
                        # Plot vertical lines for top eigenvalues
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
                        st.subheader("Top 4 Eigenvalues")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(len(eigenvalues)), eigenvalues, color='green')
                        ax.set_xticks(range(len(eigenvalues)))
                        ax.set_xticklabels([f'λ{i+1}' for i in range(len(eigenvalues))])
                        ax.set_ylabel('Eigenvalue')
                        ax.set_title('Top 4 Eigenvalues')
                        ax.grid(True, axis='y')
                        
                        for i, v in enumerate(eigenvalues):
                            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # Eigenvector statistics table
                    st.subheader("Eigenvector Statistics")
                    stats_df = analyzer.compute_eigenvector_stats(eigenvectors)
                    formatted_stats = stats_df.style.format({
                        'Mean': '{:.3f}',
                        'Std': '{:.3f}',
                        'Skewness': '{:.3f}',
                        'Kurtosis': '{:.3f}',
                        'Min': '{:.3f}',
                        'Max': '{:.3f}',
                        'Median': '{:.3f}'
                    })
                    st.dataframe(formatted_stats)
                    
                    # Top 4 eigenvectors histograms
                    st.subheader("Top 4 Eigenvectors")
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    axes = axes.ravel()
                    
                    for i in range(4):
                        sns.histplot(
                            eigenvectors[:, i],
                            bins=50,
                            ax=axes[i],
                            kde=True,
                            color=f'C{i}'
                        )
                        axes[i].set_title(f'Eigenvector {i+1}')
                        axes[i].set_xlabel('Values')
                        axes[i].set_ylabel('Frequency')
                        axes[i].grid(True)
                        
                        # Display statistical moments on the plot
                        moments_text = (
                            f'μ={np.mean(eigenvectors[:, i]):.3f}\n'
                            f'σ={np.std(eigenvectors[:, i]):.3f}\n'
                            f'Skew={stats.skew(eigenvectors[:, i]):.3f}\n'
                            f'Kurt={stats.kurtosis(eigenvectors[:, i]):.3f}'
                        )
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
                    sns.histplot(data=top_stocks_df, x='DlyCap', bins=30, ax=ax, color='purple')
                    ax.set_xlabel('Market Cap')
                    ax.set_ylabel('Count')
                    ax.set_title(f'Top {num_stocks} Stocks by Market Cap')
                    ax.grid(True)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Metrics
                    col5, col6 = st.columns(2)
                    with col5:
                        st.metric(
                            "Matrix Size",
                            f"{ret_matrix[:num_stocks].shape[0]}×{ret_matrix[:num_stocks].shape[1]}"
                        )
                        # Calculate Total Variance Explained
                        total_variance = (sum(eigenvalues) / np.trace(ret_matrix[:num_stocks] @ ret_matrix[:num_stocks].T / ret_matrix[:num_stocks].shape[1])) * 100
                        st.metric(
                            "Total Variance Explained",
                            f"{total_variance:.2f}%"
                        )
                    
                    with col6:
                        process_time = time.time() - start_time
                        st.metric("Processing Time", f"{process_time:.3f}s")
                        st.metric(
                            "Average Market Cap",
                            f"${top_stocks_df['DlyCap'].mean():,.2f}M"
                        )
                    
                    # Optional: Show stock details
                    if st.checkbox("Show Stock Details"):
                        st.dataframe(top_stocks_df[['Ticker', 'DlyCap']].reset_index(drop=True))
            
            else:
                st.error("No data available for the selected period.")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check the Parquet file and data format.")

if __name__ == "__main__":
    main()