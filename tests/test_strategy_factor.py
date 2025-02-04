import pandas as pd
import sys, os
sys.path.insert(0, "../")

from marketdata import datahandler as dh
from strategy.strategyA import StrategyA
from myenums.market_data_type import MarketDataType
from marketdata.data_adapter import CounterDataFetcher
from marketdata.counters.counterA import CounterAAdapter
from feature.feature_builder import FeatureBuilder
from factor.factor_builder import FactorBuilder
from factor.factor_loader import load_factors_from_directory
from utils.data_input import *

# 动态加载所有因子模块
factors_dir = os.path.join("../", "factor/factors")

# 加载factor/factors下的因子
load_factors_from_directory(os.path.join(factors_dir, ""), "factor/factors")


featureBuilder = FeatureBuilder()
config_filepath = r'C:\Users\12552\PycharmProjects\FlowDataTradeSystem\factor\factors_config.yml'
factorBuilder = FactorBuilder(featureBuilder, config_filepath)
strategy = StrategyA(featureBuilder, factorBuilder)

snap_dh = dh.SnapshotDataHandler()
snap_dh.subscribe(strategy.on_quote)
td_dh = dh.TransactionDataHandler()
td_dh.subscribe(strategy.on_transaction)


# def test_snap_feature():
#     for data in load_snap_info(data_fetcher, './data/snap.csv'):
#         print("-"*30)
#         if data.data_type == MarketDataType.Snapshot:
#             snap_dh.publish(None, data)

def test_trade_feature():
    index = 0
    for data in load_trade_info(data_fetcher, './data/trade.csv', max_num=100):
        index += 1
        if data.data_type == MarketDataType.Transaction:
            td_dh.publish(None, data)
        if index % 10 == 0:
            strategy.feature_builder.build_transaction_features(pd.to_datetime('2024-01-01'))
    # print(strategy.feature_builder.history_trade_feat_data)

if __name__ == "__main__":
    counter = CounterAAdapter()
    data_fetcher = CounterDataFetcher(counter)

    ## 测试构建成交特征的功能
    test_trade_feature()

    results = factorBuilder.compute_all_factors()

    print(results)

