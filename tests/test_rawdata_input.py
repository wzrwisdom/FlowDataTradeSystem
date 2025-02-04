import pandas as pd
import sys
sys.path.insert(0, "../")

from myenums.market_data_type import MarketDataType
from marketdata.data_adapter import CounterDataFetcher
from marketdata.counters.counterA import CounterAAdapter

def load_snap_info(data_fetcher, filepath):
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        if idx > 5:
            break
        print(data_fetcher.fetch_data(MarketDataType.Snapshot, row))

def load_order_info(data_fetcher, filepath):
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        if idx > 5:
            break
        print(data_fetcher.fetch_data(MarketDataType.Entrust, row))

def load_trade_info(data_fetcher, filepath):
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        if idx > 5:
            break
        print(data_fetcher.fetch_data(MarketDataType.Transaction, row))

if __name__ == "__main__":
    counter = CounterAAdapter()
    data_fetcher = CounterDataFetcher(counter)
    load_snap_info(data_fetcher, './data/snap.csv')

    # load_order_info(data_fetcher, './data/order.csv')
    #
    # load_trade_info(data_fetcher, './data/trade.csv')