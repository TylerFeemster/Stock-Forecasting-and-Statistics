import utils as u
import pandas as pd
import config
from datetime import datetime as dt
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

symbs = u.get_symbols()

request = StockBarsRequest(symbol_or_symbols=symbs, timeframe=TimeFrame(1, TimeFrameUnit.Day),
                           start=dt(year=2021, month=3, day=1))

client = StockHistoricalDataClient(
    api_key=config.API_KEY, secret_key=config.SECRET_KEY)
bars = client.get_stock_bars(request_params=request).data

df_vwap = pd.DataFrame(index=[bar.timestamp for bar in bars[symbs[0]]])
df_vol = pd.DataFrame(index=[bar.timestamp for bar in bars[symbs[0]]])
length = len(bars[symbs[0]])
for symb in symbs:
    if len(bars[symb]) == length:
        df_vwap[symb] = [bar.vwap for bar in bars[symb]]
        df_vol[symb] = [bar.volume for bar in bars[symb]]

df_vwap.to_csv('vwap.csv')
df_vol.to_csv('volume.csv')
