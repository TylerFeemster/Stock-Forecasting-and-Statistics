import utils as u
import pandas as pd
import config
from datetime import datetime as dt
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

symbs = u.get_symbols()
##
symbs=symbs[0]

request = StockBarsRequest(symbol_or_symbols=symbs, timeframe=TimeFrame(1, TimeFrameUnit.Day),
                           start=dt(year=2021, month=6, day=1))

client = StockHistoricalDataClient(
    api_key=config.API_KEY, secret_key=config.SECRET_KEY)
bars = client.get_stock_bars(request_params=request).data

print(bars[symbs])

#df = pd.DataFrame(index=[bar.timestamp for bar in bars[symbs[0]]])
#length = len(bars[symbs[0]])

#for symb in symbs:
#    array = [bar.vwap for bar in bars[symb]]
#    if len(array) == length:
#        print(symb)
#        df[symb] = array

#df.to_csv('vwap.csv')
