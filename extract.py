import utils as u
import pandas as pd
import config
from datetime import datetime as dt
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment

symbs = u.get_symbols()

request = StockBarsRequest(symbol_or_symbols=symbs, 
                           timeframe=TimeFrame(1, TimeFrameUnit.Day),
                           start=dt(year=2014, month=1, day=1), 
                           adjustment=Adjustment('all'))

client = StockHistoricalDataClient(
    api_key=config.API_KEY, secret_key=config.SECRET_KEY)
bars = client.get_stock_bars(request_params=request).data

symbols = list(bars.keys())

df_vwap = pd.DataFrame(index=[bar.timestamp for bar in bars[symbols[0]]])
df_vol = pd.DataFrame(index=[bar.timestamp for bar in bars[symbols[0]]])
df_open = pd.DataFrame(index=[bar.timestamp for bar in bars[symbols[0]]])
df_high = pd.DataFrame(index=[bar.timestamp for bar in bars[symbols[0]]])
df_low = pd.DataFrame(index=[bar.timestamp for bar in bars[symbols[0]]])
df_close = pd.DataFrame(index=[bar.timestamp for bar in bars[symbols[0]]])
length = len(bars[symbols[0]])
for symb in symbols:
    if len(bars[symb]) == length:
        df_vwap[symb] = [bar.vwap for bar in bars[symb]]
        df_vol[symb] = [bar.volume for bar in bars[symb]]
        df_open[symb] = [bar.open for bar in bars[symb]]
        df_high[symb] = [bar.high for bar in bars[symb]]
        df_low[symb] = [bar.low for bar in bars[symb]]
        df_close[symb] = [bar.close for bar in bars[symb]]

df_vwap.to_csv('./data/vwap.csv')
df_vol.to_csv('./data/volume.csv')
df_open.to_csv('./data/open.csv')
df_high.to_csv('./data/high.csv')
df_low.to_csv('./data/low.csv')
df_close.to_csv('./data/close.csv')
