import pandas as pd
import sys
sys.path.insert(0, "../")

from marketdata import datahandler as dh
from strategy.strategyA import StrategyA
from myenums.market_data_type import MarketDataType
from marketdata.data_adapter import CounterDataFetcher
from marketdata.counters.counterA import CounterAAdapter
from feature.feature_builder import FeatureBuilder
from utils.data_input import *

featureBuilder = FeatureBuilder()
strategy = StrategyA(featureBuilder, None)
snap_dh = dh.SnapshotDataHandler()
snap_dh.subscribe(strategy.on_quote)
en_dh = dh.EntrustDataHandler()
en_dh.subscribe(strategy.on_entrust)
td_dh = dh.TransactionDataHandler()
td_dh.subscribe(strategy.on_transaction)

def test_snap_feature():
    for data in load_snap_info(data_fetcher, './data/snap.csv'):
        print("-"*30)
        if data.data_type == MarketDataType.Snapshot:
            snap_dh.publish(None, data)

def test_trade_feature():
    for data in load_trade_info(data_fetcher, './data/trade.csv'):
        print("-"*30)
        if data.data_type == MarketDataType.Transaction:
            td_dh.publish(None, data)

    # print(strategy.feature_builder.trade_buffer)
    # print(strategy.feature_builder.cancel_buffer)
    strategy.feature_builder.build_transaction_features(pd.to_datetime('2024-01-01'))
    print(strategy.feature_builder.history_trade_feat_data)

def test_entrust_feature():
    for data in load_order_info(data_fetcher, './data/order.csv'):
        print("-" * 30)
        if data.data_type == MarketDataType.Entrust:
            en_dh.publish(None, data)

    print(strategy.feature_builder.entrust_buffer)
    # print(strategy.feature_builder.cancel_buffer)
    strategy.feature_builder.build_entrust_features(pd.to_datetime('2024-01-01'))
    print(strategy.feature_builder.history_entrust_feat_data)


def test_cancel_feature():
    for data in load_trade_info(data_fetcher, './data/trade.csv'):
        if data.data_type == MarketDataType.Transaction:
            td_dh.publish(None, data)
    for data in load_order_info(data_fetcher, './data/order.csv'):
        if data.data_type == MarketDataType.Entrust:
            en_dh.publish(None, data)
    print("-" * 30)
    print(strategy.feature_builder.cancel_buffer)
    strategy.feature_builder.build_cancel_features(pd.to_datetime('2024-01-01'))
    print(strategy.feature_builder.history_cancel_feat_data)

if __name__ == "__main__":
    counter = CounterAAdapter()
    data_fetcher = CounterDataFetcher(counter)

    ## 测试构建快照特征的功能
    test_snap_feature()

    # ## 测试构建成交特征的功能
    # test_trade_feature()
    #
    # ## 测试构建委托特征的功能
    # test_entrust_feature()
    #
    # ## 测试构建撤单特征的功能
    # test_cancel_feature()