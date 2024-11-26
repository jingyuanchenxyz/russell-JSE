import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy import stats
import os

# Disable matplotlib's interactive mode
plt.ioff()

# Configure Streamlit page
st.set_page_config(page_title="Market Cap Analysis", layout="wide")

@st.cache_data
def load_parquet(file_path):
    """Load and cache parquet data"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        st.error(f"Error loading parquet file: {str(e)}")
        return None

class MarketCapAnalyzer:
    def __init__(self, parquet_path):
        self.parquet_path = parquet_path
        self.df = None
        self.dates = None
        self._initialize_data()
    
    def _initialize_data(self):
        """Initialize data and dates"""
        self.df = load_parquet(self.parquet_path)
        if self.df is not None:
            self.df['DlyCalDt'] = pd.to_datetime(self.df['DlyCalDt'])
            self.dates = self.df['DlyCalDt'].dt.date.unique()
            self.dates.sort()
    
    def get_window_data(self, end_date, lookback_days):
        """Get return matrix for specified window"""
        if self.df is None:
            return None, None, None
            
        end_date = pd.to_datetime(end_date)
        mask = self.df['DlyCalDt'] <= end_date
        df_window = self.df[mask].copy()
        
        # Get latest trading days
        latest_days = df_window['DlyCalDt'].unique()
        latest_days.sort()
        if len(latest_days) < lookback_days:
            return None, None, None
            
        window_days = latest_days[-lookback_days:]
        df_window = df_window[df_window['DlyCalDt'].isin(window_days)]
        
        # Clean and prepare data
        df_window['DlyRet'] = pd.to_numeric(df_window['DlyRet'], errors='coerce')
        df_window['DlyCap'] = pd.to_numeric(df_window['DlyCap'], errors='coerce')
        df_window = df_window.drop_duplicates(subset=['DlyCalDt', 'Ticker'])
        
        # Create returns matrix
        returns_matrix = df_window.pivot(index='DlyCalDt', columns='Ticker', values='DlyRet')
        returns_matrix = returns_matrix.dropna(axis=1)
        
        # Get market caps
        last_day = window_days[-1]
        market_caps = df_window[
            (df_window['DlyCalDt'] == last_day) & 
            (df_window['Ticker'].isin(returns_matrix.columns))
        ][['Ticker', 'DlyCap']].dropna()
        
        market_caps = market_caps.sort_values('DlyCap', ascending=False)
        
        # Prepare final returns matrix
        Y = returns_matrix[market_caps['Ticker']].values.T
        Y = Y - np.mean(Y, axis=1, keepdims=True)
        
        return Y, market_caps, returns_matrix.columns.tolist()

    @staticmethod
    def analyze_market_cap_ranked(Y, num_stocks, top_k=4):
        """Compute eigendecomposition for top stocks"""
        Y_subset = Y[:num_stocks]
        U, S, _ = np.linalg.svd(Y_subset, full_matrices=False)
        eigenvalues = (S**2) / Y_subset.shape[1]
        eigenvectors = U[:, :top_k]
        
        # Normalize first eigenvector
        eigenvectors[:, 0] = eigenvectors[:, 0] - np.mean(eigenvectors[:, 0])
        
        return eigenvalues[:top_k], eigenvectors

    @staticmethod
    def compute_eigenvector_stats(eigenvectors):
        """Compute statistics for eigenvectors"""
        return pd.DataFrame([
            {
                'Eigenvector': f'EV{i+1}',
                'Mean': np.mean(evec),
                'Std': np.std(evec),
                'Skewness': stats.skew(evec),
                'Kurtosis': stats.kurtosis(evec),
                'Min': np.min(evec),
                'Max': np.max(evec),
                'Median': np.median(evec)
            }
            for i, evec in enumerate(eigenvectors.T)
        ])

def plot_eigenvalue_spectrum(eigenvalues, all_eigenvalues):
    """Plot eigenvalue spectrum"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(all_eigenvalues, bins=50, stat='density', kde=True, 
                color='blue', alpha=0.6, ax=ax)
    
    for i, ev in enumerate(eigenvalues):
        ax.axvline(ev, color=f'C{i}', linestyle='--', 
                  label=f'λ{i+1}: {ev:.3f}')
    
    ax.set_xlabel('Eigenvalues')
    ax.set_ylabel('Density')
    ax.set_title('Eigenvalue Distribution')
    ax.legend()
    ax.grid(True)
    return fig

def plot_top_eigenvalues(eigenvalues):
    """Plot top eigenvalues bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(eigenvalues)), eigenvalues, color='green')
    ax.set_xticks(range(len(eigenvalues)))
    ax.set_xticklabels([f'λ{i+1}' for i in range(len(eigenvalues))])
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Top 4 Eigenvalues')
    ax.grid(True, axis='y')
    
    for i, v in enumerate(eigenvalues):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    return fig

def main():
    st.title("Market Cap Ranked Eigenvector Analysis")
    
    # Initialize analyzer
    analyzer = MarketCapAnalyzer('r3000hist.parquet')
    
    if analyzer.df is None or len(analyzer.dates) == 0:
        st.error("Could not load data. Please check the Parquet file.")
        return
    
    # Input controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_date = st.date_input(
            "Select Analysis Date",
            value=analyzer.dates[-1],
            min_value=analyzer.dates[0],
            max_value=analyzer.dates[-1]
        )
        
        if selected_date not in analyzer.dates:
            nearest_date = min(analyzer.dates, key=lambda x: abs(x - selected_date))
            st.info(f"Using nearest trading day: {nearest_date}")
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
        step=100
    )
    
    # Process data
    with st.spinner('Processing data...'):
        ret_matrix, market_caps, all_stocks = analyzer.get_window_data(
            selected_date,
            lookback
        )
        
        if ret_matrix is None:
            st.error("Insufficient data for the selected period.")
            return
            
        eigenvalues, eigenvectors = analyzer.analyze_market_cap_ranked(
            ret_matrix,
            num_stocks
        )
        
        # Display results in tabs
        tab1, tab2 = st.tabs(["Analysis", "Details"])
        
        with tab1:
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Eigenvalue Spectrum")
                _, S, _ = np.linalg.svd(ret_matrix[:num_stocks], full_matrices=False)
                all_eigenvalues = (S**2) / ret_matrix.shape[1]
                fig = plot_eigenvalue_spectrum(eigenvalues, all_eigenvalues)
                st.pyplot(fig)
                plt.close()
            
            with col4:
                st.subheader("Top 4 Eigenvalues")
                fig = plot_top_eigenvalues(eigenvalues)
                st.pyplot(fig)
                plt.close()
            
            # Statistics
            st.subheader("Eigenvector Statistics")
            stats_df = analyzer.compute_eigenvector_stats(eigenvectors)
            st.dataframe(stats_df.style.format({col: '{:.3f}' for col in stats_df.columns 
                                              if col != 'Eigenvector'}))
        
        with tab2:
            st.subheader(f"Top {num_stocks} Stocks by Market Cap")
            st.dataframe(market_caps[['Ticker', 'DlyCap']].head(num_stocks))
            
            # Metrics
            col5, col6 = st.columns(2)
            with col5:
                st.metric("Number of Stocks", num_stocks)
                total_var = (sum(eigenvalues) / np.trace(
                    ret_matrix[:num_stocks] @ ret_matrix[:num_stocks].T / 
                    ret_matrix[:num_stocks].shape[1])) * 100
                st.metric("Total Variance Explained", f"{total_var:.2f}%")
            
            with col6:
                st.metric("Average Market Cap", 
                         f"${market_caps['DlyCap'].head(num_stocks).mean():,.2f}M")

if __name__ == "__main__":
    main()