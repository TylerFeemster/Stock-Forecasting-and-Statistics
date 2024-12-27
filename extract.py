import utils as u
import pandas as pd
import config
from datetime import datetime as dt
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment

symbs = u.all_symbols()

request = StockBarsRequest(symbol_or_symbols=symbs, 
                           timeframe=TimeFrame(1, TimeFrameUnit.Day),
                           start=dt(year=2016, month=1, day=1),
                           adjustment=Adjustment('all'))

client = StockHistoricalDataClient(
    api_key=config.API_KEY, secret_key=config.SECRET_KEY)
bars = client.get_stock_bars(request_params=request).data

symbols = list(bars.keys())

types = ['vwap', 'volume', 'open', 'close', 'high', 'low']
type_dict = {t: [] for t in types}
for symb in symbols:
    times = [bar.timestamp for bar in bars[symb]]
    for t in types:
        series = pd.Series([getattr(bar, t) for bar in bars[symb]], index=times, name=symb)
        series.index = pd.to_datetime(series.index)
        type_dict[t].append(series)

for t in types:
    pd.concat(type_dict[t], axis=1).to_csv(f'./data/{t}.csv')
