import cvxpy as cp
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

def all_symbols():
    df = pd.read_csv('./data/S&P 500 Historical Composition.csv')
    df['tickers'] = df['tickers'].apply(lambda x: x.split(','))
    df.index = pd.to_datetime(df['date'])
    df.drop('date', axis=1, inplace=True)

    all_stocks = set()
    for stocks in df['tickers'].loc[df.index > '2016-01-01']:
        for stock in stocks:
            all_stocks.add(stock)
    
    return list(all_stocks)

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
    log_diff_df = log_df.diff().loc[log_df.index[1]:]
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

#--------------------------------
# Seasonal Decomposition
#--------------------------------

class Decomposition:
    def __init__(self, data: pd.DataFrame, freq = 21):
        trend_array = []
        seasonal_array = []
        residual_array = []
        self.data = data
        for stock in data.columns:
            relevant_data = data[stock].dropna()
            if len(relevant_data) < 100:
                continue
            decomposition = sm.tsa.seasonal_decompose(relevant_data, model='additive', period=freq)
            trend_array.append(pd.Series(decomposition.trend, index=relevant_data.index, name=stock))
            seasonal_array.append(
                pd.Series(decomposition.seasonal, index=relevant_data.index, name=stock))
            residual_array.append(
                pd.Series(decomposition.resid, index=relevant_data.index, name=stock))
        self.trend = pd.concat(trend_array, axis=1)
        self.seasonal = pd.concat(seasonal_array, axis=1)
        self.resid = pd.concat(residual_array, axis=1)

        self.data.sort_index(inplace=True)
        self.trend.sort_index(inplace=True)
        self.seasonal.sort_index(inplace=True)
        self.resid.sort_index(inplace=True)

    def deseasonalized(self):
        return self.trend + self.resid
    
    def weighted_deseasonalized(self, weight = 2):
        return weight * self.trend + self.resid

    def plot(self, symbol = 'AAPL'):
        fig, ax = plt.subplots(4, 1, figsize=(10, 10))
        data = self.data[symbol].dropna()
        ax[0].plot(data.index.astype(int), data)
        ax[0].set_title('Original')
        trend = self.trend[symbol].dropna()
        ax[1].plot(trend.index.astype(int), trend)
        ax[1].set_title('Trend')
        seasonal = self.seasonal[symbol].dropna()
        ax[2].plot(seasonal.index.astype(int), seasonal)
        ax[2].set_title('Seasonal')
        residual = self.resid[symbol].dropna()
        ax[3].plot(residual.index.astype(int), residual)
        ax[3].set_title('Residual')
        fig.subplots_adjust(hspace=0.5)
        plt.show()


#--------------------------------
# Convex Optimization
#--------------------------------

def quadratic_form(data: pd.DataFrame):
    Q = data.cov().to_numpy()
    return Q

def returns(data: pd.DataFrame):
    r = data.mean(axis=0).to_numpy()
    return r

def problem_statement(data: pd.DataFrame, reweighted = False, beta = 1e-1):
    if reweighted:
        reweighted_data = Decomposition(data).weighted_deseasonalized()
        Q = quadratic_form(reweighted_data)
        r = returns(data[reweighted_data.columns])
    else:
        Q = quadratic_form(data)
        r = returns(data)

    p = cp.Variable(Q.shape[0])
    constraints = [
        cp.sum(p) == 1,
        p >= 0
    ]

    # U(p; beta) = returns - beta * trend-weighted risk
    function = r.transpose() @ p - beta * cp.quad_form(p, Q, assume_PSD=True)
    objective = cp.Maximize(function)
    return cp.Problem(objective, constraints), p

#--------------------------------
# Backtesting
#--------------------------------

