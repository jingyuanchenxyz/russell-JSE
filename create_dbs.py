import sqlite3
import pandas as pd
import os

def create_russell3000_table():
    print("Creating/Updating Russell 3000 database...")
    
    conn = sqlite3.connect('market_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS russell3000 (
        PERMNO INTEGER,
        HdrCUSIP TEXT,
        Ticker TEXT,
        PERMCO INTEGER,
        DlyCalDt DATE,
        DlyPrc REAL,
        DlyCap REAL,
        DlyRet REAL,
        PRIMARY KEY (PERMNO, DlyCalDt)
    )
    ''')
    
    print("Creating indices...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_russell_date ON russell3000(DlyCalDt)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_russell_ticker ON russell3000(Ticker)')
    
    csv_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'russell-JSE', 'wrds_export', 'r3000hist.csv')
    
    print(f"Looking for CSV file at: {csv_path}")
    if os.path.exists(csv_path):
        print("Found CSV file. Loading data...")
        
        chunksize = 100000
        chunks = pd.read_csv(csv_path, chunksize=chunksize)
        
        total_rows = 0
        for i, chunk in enumerate(chunks):
            chunk.to_sql('russell3000', conn, if_exists='append' if i > 0 else 'replace', index=False)
            total_rows += len(chunk)
            print(f"Processed {total_rows:,} rows...")
            
        print("\nCreating final indices...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_russell_date ON russell3000(DlyCalDt)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_russell_ticker ON russell3000(Ticker)')
        
        conn.commit()
        print("Database creation completed!")
        
        cursor.execute("SELECT COUNT(*) FROM russell3000")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(DISTINCT Ticker) FROM russell3000")
        tickers = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(DlyCalDt), MAX(DlyCalDt) FROM russell3000")
        date_range = cursor.fetchone()
        
        print("\nDatabase Statistics:")
        print("-" * 50)
        print(f"Total records: {count:,}")
        print(f"Unique tickers: {tickers:,}")
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        
    else:
        print(f"Error: Could not find CSV file at {csv_path}")
        print("Please ensure the CSV file is in the correct location.")
        print("\nExpected path structure:")
        print("~/Desktop/russell-JSE/wrds_export/r3000hist.csv")
    
    conn.close()

if __name__ == "__main__":
    create_russell3000_table()