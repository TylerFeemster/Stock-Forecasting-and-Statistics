import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller

# current S&P 500 companies
def get_symbols():
    df = pd.read_csv('./data/s&p500.csv')
    array = df['Symbol'].to_numpy()
    return list(array)

# get daily stock data by type
def get_stock_data(type = 'vwap') -> pd.DataFrame:
    match type:
        case 'vwap':
            df = pd.read_csv('./data/vwap.csv')
        case 'high':
            df = pd.read_csv('./data/high.csv')
        case 'low':
            df = pd.read_csv('./data/low.csv')
        case 'close':
            df = pd.read_csv('./data/close.csv')
        case 'open':
            df = pd.read_csv('./data/open.csv')
        case 'volume':
            df = pd.read_csv('./data/volume.csv')

    # correct index
    df.index = list(df['Unnamed: 0'])
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period('D')

    return df

# get total daily trading money
def get_total_money() -> pd.DataFrame:
    return get_stock_data('vwap') * get_stock_data('volume')


# log-differenced data
def log_diff(data: pd.DataFrame) -> pd.DataFrame:
    log_df = data.apply(lambda x: np.log(x+1e-6))
    log_diff_df = log_df.diff().dropna()
    return log_diff_df

# -------------------------------------
# Statistical modeling
# -------------------------------------

def is_nonstationary(data, pvalue = 0.01, display = False):
    result = adfuller(data)
    if display:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
    if result[1] < pvalue:
        return False
    return result[0:2]

def check_stationarity(data : pd.DataFrame, pvalue = 0.01):
    bad_columns = []
    adfs = []
    ps = []
    for col in tqdm(data.columns):
        result = is_nonstationary(data[col], pvalue)
        if result:
            bad_columns.append(col)
            adfs.append(result[0])
            ps.append(result[1])

    if len(bad_columns) == 0:
        print(f'All columns are stationary (threshold of p = {pvalue})')
    else:
        print(f'Non-stationary Columns (threshold of p = {pvalue})')
        for col , adf, p in zip(bad_columns, adfs, ps):
            print(f'{col:7}: ADF = {adf}, p-value = {p}')
