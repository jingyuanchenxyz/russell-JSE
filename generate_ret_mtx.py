import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
data_dict = {}
cnt = 0
with open('/Users/jingyuanchen/Desktop/russell-JSE/wrds_export/r3000hist.csv', mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if not row['DlyCalDt'] or not row['PERMNO'] or not row['DlyCap'] or not row['DlyRet']:
            continue
        date = row['DlyCalDt']
        stock_symbol = row['PERMNO']
        market_cap = float(row['DlyCap'])
        stock_ret = float(row['DlyRet'])
        if date not in data_dict:
            data_dict[date] = []
            
        data_dict[date].append((stock_symbol, market_cap, stock_ret))


start_date = '2021-07-01'
end_date = '2021-12-31'
date_range = pd.date_range(start=start_date, end=end_date)
all_stocks = []

for stock_data in data_dict[end_date]:
    stock_symbol = stock_data[0]
    market_cap = stock_data[1]
    all_stocks.append((stock_symbol, market_cap))
    
all_stocks = sorted(all_stocks, key=lambda x: x[1], reverse=True)
sorted_stock_symbols = [stock[0] for stock in all_stocks]
str_to_index = {s: i for i, s in enumerate(sorted_stock_symbols)}

ret_matrix = np.full((len(all_stocks), len(date_range)), np.nan)

for j, date in enumerate(date_range):
    str_date = date.strftime('%Y-%m-%d')
    if str_date in data_dict:
        for stock_data in data_dict[str_date]:
            stock_symbol = stock_data[0]
            stock_ret = stock_data[2]
            if stock_symbol in sorted_stock_symbols:
                i = str_to_index.get(stock_symbol)
                if i is None:
                    continue
                ret_matrix[i, j] = stock_ret

ret_df = pd.DataFrame(ret_matrix, index=all_stocks, columns=date_range)
ret_df = ret_df.dropna(how='all', axis=1)
ret_df = ret_df.dropna(how='any', axis=0)
csv_filename = 'ret_matrix.csv'
ret_df.to_csv(csv_filename, encoding='utf-8')
    
