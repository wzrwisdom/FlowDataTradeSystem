# -*- coding: UTF-8 -*-
import kungfu.yijinjing.time as kft
from pykungfu import wingchun as wc
import yaml
import pandas as pd
import sys, os
basedir = r"C:\Users\wangzirui_cx\PycharmProjects\KungfuTest"
sys.path.insert(0, basedir)
from kungfu.wingchun.constants import *
from FlowDataTradeSystem.feature.feature_builder import FeatureBuilder
from FlowDataTradeSystem.factor.factor_builder import FactorBuilder
from FlowDataTradeSystem.factor.factor_loader import load_factors_from_directory
from FlowDataTradeSystem.model.model_base import ModelBase
from FlowDataTradeSystem.tests.utils.prepareData import (
    prepareTrade, prepareOrder, prepareQuote
)

# from FlowDataTradeSystem.factor.factor_builder import myprint
# 期货
# SOURCE = "ctp"
# ACCOUNT = "089270"
# tickers = ["rb2001","rb2003"]
# VOLUME = 2
# EXCHANGE = Exchange.SHFE

# 股票柜台
SOURCE = "otc"
# 要链接的账户
ACCOUNT = "single-OTC000082645-tcp223701242299709"
# 准备订阅的标的
tickers = ["000001"]
# 下单数量s
VOLUME = 200
# 标的对应的交易所
EXCHANGE = Exchange.SSE

# 动态加载所有因子模块
factors_dir = rf"{basedir}\FlowDataTradeSystem\factor\factors"

# 加载factor/factors下的因子
load_factors_from_directory(os.path.join(factors_dir, ""), "factor/factors")

# featureBuilder = FeatureBuilder()
# config_filepath = rf'{basedir}\FlowDataTradeSystem\factor\factors_config.yml'
# factorBuilder = FactorBuilder(featureBuilder, config_filepath)

featBuilderDict = {}
facBuilderDict = {}
config_filepath = rf'{basedir}\FlowDataTradeSystem\factor\factors_config.yml'
for ticker in tickers:
    featBuilderDict[ticker] = FeatureBuilder()
    facBuilderDict[ticker] = FactorBuilder(featBuilderDict[ticker], config_filepath)

def get_model_dict(factors_name):
    # model_dict = ModelBase.create_model("lin_model", "")
    model_dict = {}
    folder_path = r"C:\Users\wangzirui_cx\PycharmProjects\KungfuTest\FlowDataTradeSystem\tests\data"
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)) and 'lin_model_allfac_test' in filename:
            symbol = filename[-9:]
            model = ModelBase.create_model("lin_model", os.path.join(folder_path, filename), factors_name)
            model_dict[symbol] = model
    return model_dict

with open(config_filepath, 'r') as f:
    factors_info = yaml.load(f, Loader=yaml.FullLoader)
factors_name = list(factors_info.keys())
model_dict = get_model_dict(factors_name)

par_dict = {
    'preprocess_filepath': r"C:\Users\wangzirui_cx\PycharmProjects\KungfuTest\FlowDataTradeSystem\tests\data\data_process_params\{}.yml",
    'model_dict': model_dict,
    'judge_col': 'spread1',
    'no_winsorize_factors': ['DeriPBid1', 'DeriPAsk1'],
    'buy_threshold': 0.00029436267446151,
    'sell_threshold': -0.00019256103582417826,
    'close_buy_threshold': -0.00015317740950684968,
    'close_sell_threshold': 0.0002524555690396821,
    'vol': 10000
}


# 启动前回调，添加交易账户，订阅行情，策略初始化计算等
def pre_start(context):
    context.add_account(SOURCE, ACCOUNT)
    context.subscribe(SOURCE, tickers, EXCHANGE)
    context.depth_orderbook = {}


# 启动准备工作完成后回调，策略只能在本函数回调以后才能进行获取持仓和报单
def post_start(context):
    context.log.warning("post_start")
    log_book(context, None)


# 收到快照行情时回调，行情信息通过quote对象获取
def on_quote(context, quote, location,dest):
    context.log.info("[on_quote] {}".format(quote))
    if quote.instrument_id in tickers:
        time = context.strftime(context.now())
        data = prepareQuote(quote, time)

        featureBuilder = featBuilderDict[quote.instrument_id]
        factorBuilder = facBuilderDict[quote.instrument_id]

        featureBuilder.build_snap_features(data)
        results = factorBuilder.compute_all_factors()
        symbol_ = data['symbol']
        preprocess_filepath = par_dict['preprocess_filepath'].format(symbol_)
        factors = factorBuilder.preprocess(preprocess_filepath, par_dict['judge_col'],
                                                 par_dict['no_winsorize_factors'],
                                                 factors=results.copy())
        model = par_dict['model_dict'][symbol_]
        context.log.info("[features] {}".format(featureBuilder.history_feat_dict))
        context.log.info("[factors] {}".format(factors))
        model_v = model.predict(factors)
        signal = generate_signal(par_dict, model_v)
        close_signal = generate_close_signal(par_dict, model_v)
        if (close_signal == 1) and can_place_order(context, quote.instrument_id, VOLUME, None, Side.Sell, Offset.Close):
            pass
            order_id = context.insert_order(quote.instrument_id, EXCHANGE, SOURCE, ACCOUNT, data['offer_prices'][0],
                                            VOLUME,
                                            PriceType.Limit, Side.Sell, Offset.Close)
        elif (close_signal == -1) and can_place_order(context, quote.instrument_id, VOLUME, None, Side.Buy, Offset.Close):
            pass
            order_id = context.insert_order(quote.instrument_id, EXCHANGE, SOURCE, ACCOUNT, data['bid_prices'][0],
                                            VOLUME,
                                            PriceType.Limit, Side.Buy, Offset.Close)
        elif (signal == 1) and can_place_order(context, quote.instrument_id, VOLUME, data['bid_prices'][0], Side.Buy, Offset.Open):
            pass
            order_id = context.insert_order(quote.instrument_id, EXCHANGE, SOURCE, ACCOUNT, data['bid_prices'][0],
                                            VOLUME,
                                            PriceType.Limit, Side.Buy, Offset.Open)
        elif (signal == -1) and can_place_order(context, quote.instrument_id, VOLUME, data['offer_prices'][0], Side.Sell, Offset.Open):
            pass
            order_id = context.insert_order(quote.instrument_id, EXCHANGE, SOURCE, ACCOUNT, data['offer_prices'][0],
                                            VOLUME,
                                            PriceType.Limit, Side.Sell, Offset.Open)
        context.log.info("[factor comb] (value){} (signal){} (close_signal){}".format(model_v, signal, close_signal))
        order_id = context.insert_order(quote.instrument_id, EXCHANGE, SOURCE, ACCOUNT, quote.last_price+0.1, VOLUME,
                                        PriceType.Limit, Side.Buy, Offset.Open)
        context.log.info("[order] (rid){} (ticker){}".format(order_id, quote.instrument_id))

        # 更新最新的盘口价格
        context.depth_orderbook[quote.instrument_id] = {'bid1': data['bid_prices'][0], 'ask1': data['offer_prices'][0]}
        # if order_id > 0:
        #     # 通过添加时间回调，在三秒以后撤单
        #     context.add_timer(context.now() + 3 * 1000000000, lambda ctx, event: cancel_order(ctx, order_id))


# 收到订单状态回报时回调
def on_order(context, order, location, dest):
    context.log.info("[on_order] {}".format(order))
    if (order.order_id > 0 and order.offset == Offset.Open):
        # 通过添加时间回调，在三秒以后撤单
        context.add_timer(context.now() + 3 * 1000000000, lambda ctx, event: cancel_order(ctx, order))


# 收到成交信息回报时回调
def on_trade(context, trade, location, dest):
    context.log.info("[on_trade] {}".format(trade))



def on_transaction(context, transaction, location, dest):
    context.log.info('[on_transaction] {}'.format(transaction))
    featureBuilder = featBuilderDict[transaction.instrument_id]
    trade = prepareTrade(transaction)
    featureBuilder.add_transaction(trade)

def on_entrust(context, entrust, location, dest):
    context.log.info('[on_entrust] {}'.format(entrust))
    featureBuilder = featBuilderDict[entrust.instrument_id]
    order = prepareOrder(entrust)
    if order['datetime'].time() < pd.to_datetime('09:30:00').time():
        featureBuilder.entrust_dict_by_appl_seq[order['appl_seq_num']] = order
        return
    featureBuilder.add_entrust(order)


# 策略退出前方法，仍然可以获取持仓和报单
def pre_stop(context):
    context.log.info("[before strategy stop]")


# 策略进程退出前方法
def post_stop(context):
    context.log.info("[before process stop]")


def can_place_order(context, instrument_id, volume, price, side, offset):
    if offset == Offset.Open:
        return has_sufficient_balance(context, instrument_id, volume, price, side)
    elif offset == Offset.Close:
        return has_sufficient_position(context, instrument_id, volume, side)

# 判断是否有充足的仓位
def has_sufficient_position(context, instrument_id, volume, side):
    account_uid = context.get_account_uid(SOURCE, ACCOUNT)
    position_key = wc.utils.hash_instrument(account_uid, EXCHANGE, instrument_id)
    position_volume = 0
    if side == Side.Sell:
        position = context.get_account_book(SOURCE, ACCOUNT).long_positions[position_key]
        position_volume = position.volume
    elif side == Side.Buy:
        position = context.get_account_book(SOURCE, ACCOUNT).short_positions[position_key]
        position_volume = position.volume
    if position_volume > volume:
        return True
    else:
        return False


# 判断是否有充足的资金
def has_sufficient_balance(context, symbol, volume, price, side):
    need_amount = volume * price
    book = context.get_account_book(SOURCE, ACCOUNT)
    min_commission = 0.001
    if book.asset.avail > need_amount * (1 + min_commission):
        return True
    else:
        return False


# 自定义函数
# 账户中资金/持仓情况
def log_book(context, event):
    context.account_book = context.get_account_book(SOURCE, ACCOUNT)

    context.log.warning("账户资金组合信息 {}".format(context.account_book.asset))

    # 账户中多头持仓数据
    long_position = context.account_book.long_positions
    for key in long_position:
        pos = long_position[key]
        context.log.info("多头持仓数据 (instrument_id){} (volume){} (yesterday_volume){}".format(pos.instrument_id,pos.volume,pos.yesterday_volume))

# 自定义撤单回调函数
def cancel_order(context, order):
    order_id = order.order_id
    action_id = context.cancel_order(order_id)
    if action_id > 0:
        context.log.info("[cancel order] (action_id){} (rid){} ".format(action_id, order_id))


# 自定义撤单并对未成交的进行补充下单的回调函数
def cancel_reorder(context, order):
    order_id = order.order_id
    action_id = context.cancel_order(order_id)
    if action_id > 0:
        context.log.info("[cancel and reorder] (action_id){} (rid){} ".format(action_id, order_id))
    if order.side == Side.Buy:
        price = context.depth_orderbook[order.instrument_id]['bid1']
        order_id = context.insert_order(order.instrument_id, EXCHANGE, SOURCE, ACCOUNT, price, order.volume_left,
                                        PriceType.Limit, Side.Buy, Offset.Open)
    elif order.side == Side.Sell:
        price = context.depth_orderbook[order.instrument_id]['ask1']
        order_id = context.insert_order(order.instrument_id, EXCHANGE, SOURCE, ACCOUNT, price, order.volume_left,
                                        PriceType.Limit, Side.Sell, Offset.Open)

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