import time

from FlowDataTradeSystem.strategy.strategy import Strategy
from loguru import logger as log
import pandas as pd
class StrategyC(Strategy):
    def __init__(self, feature_builderDict, factor_builderDict, context=None):
        super(StrategyC, self).__init__()
        self.feature_builderDict = feature_builderDict
        self.factor_builderDict = factor_builderDict
        self.context = context
        self.context['comb_time'] = None

    def on_quote(self, context, data):
        feature_builder = self.feature_builderDict[data['symbol']]
        factor_builder = self.factor_builderDict[data['symbol']]
        # log.info("Accept snapshot info and build features")
        # time1 = time.perf_counter()
        feature_builder.build_snap_features(data)
        if data['markettype'] == 'Future':
            factor_builder.compute_all_factors()


        # time2 = time.perf_counter()
        # time3 = time.perf_counter()
        # elapsed_time = time3 - time2
        # # log.info(f"计算特征耗时{elapsed_time:.4f}")
        # factor_builder.compute_all_factors()
        # time3 = time.perf_counter()
        # elapsed_time = time3 - time2
        # # log.info(f"计算因子耗时{elapsed_time:.4f}")
        #
        # if data['markettype'] == 'Fund':
        #     # symbol_ = data['symbol']
        #     # fund_results = self.factor_builderDict[symbol_].get_results()
        #     # log.info(fund_results)
        #     symbol_ = data['symbol']
        #     future_symbol = 'IF'
        #     fund_results = self.factor_builderDict[symbol_].get_results()
        #     future_results = self.factor_builderDict[future_symbol].get_results()
        #     results = future_results.copy()
        #     for key, value in fund_results.items():
        #         results[key+'_fund'] = value
        #     # log.info(results)
        #     preprocess_filepath = self.context["preprocess_filepath"].format(symbol_)
        #     factors = factor_builder.preprocess(preprocess_filepath, self.context['judge_col'], self.context['no_winsorize_factors'], factors=results.copy())
        #     # log.info("[After preprocess]: {}".format(factors))
        #     if factors is None:
        #         return
        #     else:
        #         model = self.context['model_dict'][symbol_]
        #         model_v, features = model.predict(factors)
        #         if data.data['datetime'].time() == pd.Timestamp("09:48:57").time():
        #             log.info(features)
        #     #     # model_v = 1.0
        #     log.info(model_v)
            # # self.factor_builder.add_facComb(symbol_, model_v)
            # time4 = time.perf_counter()
            # elapsed_time = time4 - time3
            # log.info(f"因子预处理和组合耗时{elapsed_time:.4f}")
            # signal = generate_signal(self.context, model_v)
            # close_signal = generate_close_signal(self.context, model_v)
            # avg_price = (data['ask_prices'][0] + data['bid_prices'][0])/2
            # if signal == 1:
            #     orderid = self.buy(symbol_, avg_price, vol=self.context['vol'], close_flag=False)
            # elif signal == -1:
            #     orderid = self.sell(symbol_, avg_price, vol=self.context['vol'], close_flag=False)
            #
            # long_pos, short_pos = self.broker.position(symbol_)
            # if (long_pos > 0) and close_signal == 1:
            #     orderid = self.sell(symbol_, avg_price, vol=self.context['vol'], close_flag=True)
            # elif (short_pos > 0) and close_signal == -1:
            #     orderid = self.buy(symbol_, avg_price, vol=self.context['vol'], close_flag=True)


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

        # 更新需要进行因子组合的时刻，有逐笔委托信息的时间进行触发因子组合，触发后完成所有标的的因子组合
        if context['comb_time'] is None:
            context['comb_time'] = get_aligned_time(entrust['datetime'])
        else:
            if entrust['datetime'] > context['comb_time']:
                self.compute_factor_and_comb(context)
                context['comb_time'] = context['comb_time'] + pd.Timedelta(seconds=3)


        feature_builder.add_entrust(entrust)


def compute_factor_and_comb(self, context):
    for symbol_ in context['symbols']:
        factor_builder = self.factor_builderDict[symbol_]
        factor_builder.compute_all_factors()
        future_symbol = 'IF'
        fund_results = self.factor_builderDict[symbol_].get_results()
        future_results = self.factor_builderDict[future_symbol].get_results()
        results = future_results.copy()
        for key, value in fund_results.items():
            results[key+'_fund'] = value
        # log.info(results)
        preprocess_filepath = self.context["preprocess_filepath"].format(symbol_)
        factors = factor_builder.preprocess(preprocess_filepath, self.context['judge_col'], self.context['no_winsorize_factors'], factors=results.copy())
        # log.info("[After preprocess]: {}".format(factors))
        if factors is None:
            return

        model = self.context['model_dict'][symbol_]
        model_v, features = model.predict(factors)
        log.info(model_v)


def get_aligned_time(given_time, base_time="09:30:00"):
    base_time = pd.Timestamp(base_time)

    elapsed_time = given_time - base_time
    elapsed_ms = (given_time - base_time).total_seconds() * 1000

    interval_ms = 3000
    a = 1 if (elapsed_ms % interval_ms) > 0 else 0
    aligned_ms = (elapsed_ms // interval_ms + a) * interval_ms  # 向上取整
    aligned_time = base_time + pd.to_timedelta(aligned_ms, unit="ms")
    return aligned_time

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




