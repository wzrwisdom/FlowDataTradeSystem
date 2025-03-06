import pandas as pd
import sys, os
import yaml
sys.path.insert(0, "../../")

from FlowDataTradeSystem.marketdata import datahandler as dh
from FlowDataTradeSystem.strategy.strategyB import StrategyB
from FlowDataTradeSystem.myenums.market_data_type import MarketDataType
from FlowDataTradeSystem.marketdata.data_adapter import CounterDataFetcher
from FlowDataTradeSystem.marketdata.counters.counterFutureAndFund import CounterFutureAndFundAdapter
from FlowDataTradeSystem.feature.feature_builder import FeatureBuilder
from FlowDataTradeSystem.feature.future_feature_builder import FutureFeatureBuilder
from FlowDataTradeSystem.factor.factor_builder import FactorBuilder
from FlowDataTradeSystem.factor.factor_loader import load_factors_from_directory
from FlowDataTradeSystem.model.model_base import ModelBase
from utils.data_input import *

# 动态加载所有因子模块
factors_dir = os.path.join("../", "factor/factors")
# 加载factor/factors下的因子
load_factors_from_directory(os.path.join(factors_dir, ""), "factor/factors")
factors_dir = os.path.join("../", "factor/future_factors")
load_factors_from_directory(os.path.join(factors_dir, ""), "factor/future_factors")

def get_model_dict(factors_name):
    # model_dict = ModelBase.create_model("lin_model", "")
    model_dict = {}
    folder_path = './data/'
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)) and 'lin_model_allfac_test' in filename:
            symbol = filename[-9:]
            model = ModelBase.create_model("lin_model", os.path.join(folder_path, filename), factors_name)
            model_dict[symbol] = model
    return model_dict

# featureBuilder = FeatureBuilder()
# config_filepath = r'C:\Users\12552\PycharmProjects\FlowDataTradeSystem\factor\factors_config.yml'
# factorBuilder = FactorBuilder(featureBuilder, config_filepath)

symbols = ['510310.SH']
future_symbols = ['IF']
featBuilderDict = {}
facBuilderDict = {}
config_filepath = r'C:\Users\12552\PycharmProjects\FlowDataTradeSystem\factor\factors_config.yml'
future_config_filepath = r'C:\Users\12552\PycharmProjects\FlowDataTradeSystem\factor\future_factors_config.yml'
# for symbol in symbols:
#     featBuilderDict[symbol] = FeatureBuilder()
#     facBuilderDict[symbol] = FactorBuilder(featBuilderDict[symbol], config_filepath)

for symbol in future_symbols:
    featBuilderDict[symbol] = FutureFeatureBuilder()
    facBuilderDict[symbol] = FactorBuilder(featBuilderDict[symbol], future_config_filepath)


with open(config_filepath, 'r') as f:
    factors_info = yaml.load(f, Loader=yaml.FullLoader)
factors_name = list(factors_info.keys())
model_dict = get_model_dict(factors_name)
context = {
    'preprocess_filepath': './data/data_process_params/{}.yml',
    'model_dict': model_dict,
    'judge_col': 'spread1',
    'no_winsorize_factors': ['DeriPBid1', 'DeriPAsk1'],
    'buy_threshold': 0.00029436267446151,
    'sell_threshold': -0.00019256103582417826,
    'close_buy_threshold': -0.00015317740950684968,
    'close_sell_threshold': 0.0002524555690396821,
    'vol': 10000
}

strategy = StrategyB(featBuilderDict, facBuilderDict, context=context)

snap_dh = dh.SnapshotDataHandler()
snap_dh.subscribe(strategy.on_quote)
en_dh = dh.EntrustDataHandler()
en_dh.subscribe(strategy.on_entrust)
td_dh = dh.TransactionDataHandler()
td_dh.subscribe(strategy.on_transaction)
dh.DataHandler._registry['SnapshotDataHandler'] = snap_dh
dh.DataHandler._registry['EntrustDataHandler'] = en_dh
dh.DataHandler._registry['TransactionDataHandler'] = td_dh



# def test_snap_feature():
#     for data in load_snap_info(data_fetcher, './data/snap.csv'):
#         print("-"*30)
#         if data.data_type == MarketDataType.Snapshot:
#             snap_dh.publish(None, data)

# def test_trade_feature():
#     index = 0
#     for data in load_trade_info(data_fetcher, './data/trade.csv', max_num=100):
#         index += 1
#         if data.data_type == MarketDataType.Transaction:
#             td_dh.publish(None, data)
#         if index % 10 == 0:
#             strategy.feature_builderDict['510310.SH'].build_transaction_features(pd.to_datetime('2024-01-01'))
#     # print(strategy.feature_builder.history_trade_feat_data)

def test_trade():
    counter = CounterFutureAndFundAdapter()
    data_fetcher = CounterDataFetcher(counter)

    for data in load_future_and_fund_data(data_fetcher):
        # print(data)
        dh.DataHandler.publish_data(context=None, data=data)

# LI td_p_v_ratio td_ret_v_prod pv_corr en_b_price_std en_s_price_tsrank ATR VLI

if __name__ == "__main__":
    counter = CounterFutureAndFundAdapter()
    data_fetcher = CounterDataFetcher(counter)

    for data in load_future_and_fund_data(data_fetcher):
        # print(data)
        dh.DataHandler.publish_data(context=None, data=data)

    # ## 测试构建成交特征的功能
    # test_trade_feature()
    #
    # results = factorBuilder.compute_all_factors()
    #
    # # 模型配置
    # model_type = "lin_model"  # 可切换为 "CNN"
    # model_path = os.path.join("models", "linear_model.pkl")  # 模型文件路径
    # input_features = ["td_p_v_ratio", "volume_weighted_price"]  # 模型输入因子
    #
    # # 动态加载模型
    # model = ModelBase.create_model(model_type, model_path, input_features)
    # y_pred = model.predict(results)

