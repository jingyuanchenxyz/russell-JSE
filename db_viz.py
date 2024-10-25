import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime

def load_data(date, n_stocks=126):
    """Load data from database for a specific date"""
    conn = sqlite3.connect('market_database.db')
    query = """
    SELECT Ticker, DlyCalDt, DlyPrc, DlyCap, DlyRet
    FROM russell3000
    WHERE DlyCalDt = ?
    ORDER BY DlyCap DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(date, n_stocks))
    conn.close()
    return df

def get_date_range():
    """Get available date range in database"""
    conn = sqlite3.connect('market_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT MIN(DlyCalDt), MAX(DlyCalDt) FROM russell3000")
    min_date, max_date = cursor.fetchone()
    conn.close()
    return min_date, max_date

st.title("Russell 3000 Database Viewer")

min_date, max_date = get_date_range()

st.sidebar.header("Controls")
selected_date = st.sidebar.date_input(
    "Select Date",
    value=datetime.strptime(max_date, '%Y-%m-%d').date(),
    min_value=datetime.strptime(min_date, '%Y-%m-%d').date(),
    max_value=datetime.strptime(max_date, '%Y-%m-%d').date()
)

n_stocks = st.sidebar.slider("Number of stocks to display", 10, 500, 100)

data = load_data(selected_date, n_stocks)

st.header("Database Overview")
st.write(f"Selected date: {selected_date}")
st.write(f"Number of stocks shown: {len(data)}")

st.header("Market Cap Distribution")
fig = px.histogram(data, x='DlyCap', nbins=50, title='Market Cap Distribution')
st.plotly_chart(fig)

st.header("Top Stocks by Market Cap")
fig = px.bar(
    data.head(20), 
    x='Ticker', 
    y='DlyCap',
    title='Top 20 Stocks by Market Cap'
)
st.plotly_chart(fig)

st.header("Raw Data")
st.dataframe(data)

csv = data.to_csv(index=False)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name=f'russell3000_{selected_date}.csv',
    mime='text/csv',
)