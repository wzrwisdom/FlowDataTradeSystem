import time

from FlowDataTradeSystem.strategy.strategy import Strategy
from loguru import logger as log
import pandas as pd
class StrategyA(Strategy):
    def __init__(self, feature_builderDict, factor_builderDict, context=None):
        super(StrategyA, self).__init__()
        self.feature_builderDict = feature_builderDict
        self.factor_builderDict = factor_builderDict
        self.context = context

    def on_quote(self, context, data):
        # print(data)
        feature_builder = self.feature_builderDict[data['symbol']]
        factor_builder = self.factor_builderDict[data['symbol']]
        log.info("Accept snapshot info and build features")
        time1 = time.perf_counter()
        feature_builder.build_snap_features(data)
        time2 = time.perf_counter()
        time3 = time.perf_counter()
        elapsed_time = time3 - time2
        log.info(f"计算特征耗时{elapsed_time:.4f}")
        results = factor_builder.compute_all_factors()
        time3 = time.perf_counter()
        elapsed_time = time3 - time2
        log.info(f"计算因子耗时{elapsed_time:.4f}")
        log.info(results)

        symbol_ = data['symbol']
        preprocess_filepath = self.context["preprocess_filepath"].format(symbol_)
        factors = factor_builder.preprocess(preprocess_filepath, self.context['judge_col'], self.context['no_winsorize_factors'], factors=results.copy())
        if factors is None:
            return
        else:
            model = self.context['model_dict'][symbol_]
            model_v = model.predict(factors)
            # model_v = 1.0
        log.info(model_v)
        # self.factor_builder.add_facComb(symbol_, model_v)
        time4 = time.perf_counter()
        elapsed_time = time4 - time3
        log.info(f"因子预处理和组合耗时{elapsed_time:.4f}")
        signal = generate_signal(self.context, model_v)
        close_signal = generate_close_signal(self.context, model_v)
        avg_price = (data['ask_prices'][0] + data['bid_prices'][0])/2
        if signal == 1:
            orderid = self.buy(symbol_, avg_price, vol=self.context['vol'], close_flag=False)
        elif signal == -1:
            orderid = self.sell(symbol_, avg_price, vol=self.context['vol'], close_flag=False)

        long_pos, short_pos = self.broker.position(symbol_)
        if (long_pos > 0) and close_signal == 1:
            orderid = self.sell(symbol_, avg_price, vol=self.context['vol'], close_flag=True)
        elif (short_pos > 0) and close_signal == -1:
            orderid = self.buy(symbol_, avg_price, vol=self.context['vol'], close_flag=True)
        # print(self.feature_builder.history_feat_dict)

    def on_trade(self, context, data):
        print(data)

    def on_transaction(self, context, transaction):
        # print(transaction)
        # log.info("Accept trade info")
        feature_builder = self.feature_builderDict[transaction['symbol']]
        feature_builder.add_transaction(transaction)

    def on_order(self, context, order):
        print(order)

    def on_entrust(self, context, entrust):
        # log.info("Accept entrust info")
        feature_builder = self.feature_builderDict[entrust['symbol']]
        # 将09：30之前的逐笔委托提前导入特征构造类中
        if entrust['datetime'].time() < pd.to_datetime('09:30:00').time():
            feature_builder.entrust_dict_by_appl_seq[entrust['appl_seq_num']] = entrust
            return
        feature_builder.add_entrust(entrust)


def generate_signal(context, model_v):
    # 1代表买入 -1代表卖出
    if model_v > context['buy_threshold']:
        return 1
    elif model_v < context['sell_threshold']:
        return -1
    else:
        return 0

def generate_close_signal(context, model_v):
    if model_v < context['close_buy_threshold']:
        return 1
    elif model_v > context['close_sell_threshold']:
        return -1
    else:
        return 0




