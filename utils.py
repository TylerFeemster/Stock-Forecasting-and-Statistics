import pandas as pd

def get_symbols():
    df = pd.read_csv('./s&p500.csv')
    array = df['Symbol'].to_numpy()
    return list(array)