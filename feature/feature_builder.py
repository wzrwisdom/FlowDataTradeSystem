from sortedcontainers import SortedDict
from collections import deque
import math
from loguru import logger as log
import numpy as np
import pandas as pd

class FeatureBuilder:
    def __init__(self):
        # self.history_snap_feat_data = SortedDict()
        # self.history_trade_feat_data = SortedDict()
        # self.history_entrust_feat_data = SortedDict()
        # self.history_cancel_feat_data = SortedDict()
        self.maxlen = 1000
        self.history_feat_dict = {}
        self.history_time = deque(maxlen=self.maxlen)
        # self.history_snap_feat_data = deque(maxlen=maxlen)
        # self.history_trade_feat_data = deque(maxlen=maxlen)
        # self.history_entrust_feat_data = deque(maxlen=maxlen)
        # self.history_cancel_feat_data = deque(maxlen=maxlen)
        self.trade_buffer = deque()
        self.entrust_buffer = deque()
        self.cancel_buffer = deque()
        self.previous_snap_feat = {}
        self.previous_trade_feat = {}
        self.previous_entrust_feat = {}
        self.previous_cancel_feat = {}
        self.entrust_dict_by_appl_seq = {}

    # def get_recent_features(self, window_size, _type='snap'):
    #     if _type == 'snap':
    #         return list(self.history_snap_feat_data)[-window_size:]
    #     elif _type == 'trade':
    #         return list(self.history_trade_feat_data)[-window_size:]
    #     elif _type == 'entrust':
    #         return list(self.history_entrust_feat_data)[-window_size:]
    #     elif _type == 'cancel':
    #         return list(self.history_cancel_feat_data)[-window_size:]
    #     else:
    #         log.error(f"没有{_type}类型的特征")

    def get_recent_features(self, name, window):
        if name not in self.history_feat_dict.keys():
            log.error(f"特征类中没有{name}的数据")
        if window == None:
            return pd.Series(list(self.history_feat_dict[name]))
        return pd.Series(list(self.history_feat_dict[name])[-window:])

    def add_feature(self, timestamp, feature_info):
        if timestamp not in self.history_time:
            log.warning("特征对应的时间和 history_time中的最近时间不匹配") # 后续可以注释
        for name, value in feature_info.items():
            if name not in self.history_feat_dict.keys():
                self.history_feat_dict[name] = deque(maxlen=self.maxlen)
            self.history_feat_dict[name].append(value)
            # if name not in self.history_feat_dict.keys():
                # self.history_feat_dict[name] = np.full(self.maxlen, np.nan)
            # self.history_feat_dict[name] = np.roll(self.history_feat_dict[name], -1)  # 向左滚动数组
            # self.history_feat_dict[name][-1] = value

    # def add_snap_feature(self, timestamp, feature):
    #     self.history_snap_feat_data.append((timestamp, feature))
    #
    # def add_trade_feature(self, timestamp, feature):
    #     self.history_trade_feat_data.append((timestamp, feature))
    #
    # def add_entrust_feature(self, timestamp, feature):
    #     self.history_entrust_feat_data.append((timestamp, feature))
    #
    # def add_cancel_feature(self, timestamp, feature):
    #     self.history_cancel_feat_data.append((timestamp, feature))


    def build_snap_features(self, data):
        ##########   构建特征所需函数   ######
        def calculate_wb(bv_sum, sv_sum, last_wb):
            try:
                return (bv_sum - sv_sum) / (bv_sum + sv_sum)
            except ZeroDivisionError:
                return last_wb
        ###################################
        features = {}

        sv1_sum = data['ask_volumes'][0]
        bv1_sum = data['bid_volumes'][0]
        features.update({
            'datetime': data['datetime'],
            's1': data['ask_prices'][0], 's5': data['ask_prices'][4], 's10': data['ask_prices'][9],
            'b1': data['bid_prices'][0], 'b5': data['bid_prices'][4], 'b10': data['bid_prices'][9],
            'sv1_sum': sv1_sum, 'bv1_sum': bv1_sum,
        })
        features.update({
            'ssv1_sum': features['s1'] * features['sv1_sum'],
            'bbv1_sum': features['b1'] * features['bv1_sum'],
        })

        sv5_sum, bv5_sum = 0, 0
        ssv5_sum, bbv5_sum = 0, 0
        for i in range(0, 5):
            sv5_sum += data['ask_volumes'][i]
            bv5_sum += data['bid_volumes'][i]
            ssv5_sum += data['ask_prices'][i] * data['ask_volumes'][i]
            bbv5_sum += data['bid_prices'][i] * data['bid_volumes'][i]

        sv10_sum, bv10_sum = sv5_sum, bv5_sum
        ssv10_sum, bbv10_sum = ssv5_sum, bbv5_sum
        for i in range(5, 10):
            sv10_sum += data['ask_volumes'][i]
            bv10_sum += data['bid_volumes'][i]
            ssv10_sum += data['ask_prices'][i] * data['ask_volumes'][i]
            bbv10_sum += data['bid_prices'][i] * data['bid_volumes'][i]

        wb1 = calculate_wb(bv1_sum, sv1_sum, last_wb=features.get('wb1', 0))
        wb5 = calculate_wb(bv5_sum, sv5_sum, last_wb=features.get('wb5', 0))
        wb10 = calculate_wb(bv10_sum, sv10_sum, last_wb=features.get('wb10', 0))
        bs_avg_price = (data['ask_prices'][0] + data['bid_prices'][0]) / 2

        features.update({
            # 'close': data["last_price"],
            'sv5_sum': sv5_sum, 'bv5_sum': bv5_sum,
            'ssv5_sum': ssv5_sum, 'bbv5_sum': bbv5_sum,
            'sv10_sum': sv10_sum, 'bv10_sum': bv10_sum,
            'ssv10_sum': ssv10_sum, 'bbv10_sum': bbv10_sum,
            'wb1': wb1, 'wb5': wb5, 'wb10': wb10, 'bs_avg_price': bs_avg_price
        })

        self.history_time.append(features["datetime"])
        self.add_feature(features["datetime"], features)
        self.previous_snap_feat = features

        # 考虑到快照信息是按照给定的频率发送的，以接收到快照信息的时点为判断，进行计算成交特征以及委托特征
        self.build_transaction_features(features["datetime"])
        self.build_entrust_features(features["datetime"])
        self.build_cancel_features(features["datetime"])




    def add_transaction(self, data):
        # 将新的成交数据存入 trade_buffer 中
        if data['trade_price'] == 0.0:
            new_data = {
                'symbol': data['symbol'],
                'datetime': data['datetime'],
                'appl_seq_num': data['ask_appl_seq_num'] if data["bid_appl_seq_num"] == 0 else data['bid_appl_seq_num'],
                'side': 'S' if data["bid_appl_seq_num"] == 0 else 'B',
                'volume': data['trade_volume'],
            }
            self.cancel_buffer.append(new_data)
        else:
            self.trade_buffer.append(data)

    def add_entrust(self, data):
        # 将新的委托数据存入 entrust_buffer 中
        if data['order_type'] == 'D':
            new_data = {
                'symbol': data['symbol'],
                'datetime': data['datetime'],
                'appl_seq_num': data['appl_seq_num'],
                'side': data['side'],
                'volume': data['order_volume'],
            }
            # print(new_data)
            self.cancel_buffer.append(new_data)
        else:
            self.entrust_dict_by_appl_seq[data['appl_seq_num']] = data
            self.entrust_buffer.append(data)

    def build_transaction_features(self, datetime):
        ##########   构建特征所需函数   ######
        def calculate_ratio(numerator, denominator):
            try:
                return numerator / denominator
            except ZeroDivisionError:
                return 0

        ###################################

        if len(self.trade_buffer) == 0:
            features = {
                'datetime': datetime,
                'open': self.previous_trade_feat.get('close', None),
                'close': self.previous_trade_feat.get('close', None),
                'high': self.previous_trade_feat.get('close', None),
                'low': self.previous_trade_feat.get('close', None),
                'vwap': self.previous_trade_feat.get('vwap', None),
                'td_buy_num': 0, 'td_sell_num': 0,
                'td_buy_price': self.previous_trade_feat.get('td_buy_price', None),
                'td_sell_price': self.previous_trade_feat.get('td_sell_price', None),
                'td_buy_vol': 0, 'td_sell_vol': 0,
                # 'td_buy_price_std': 0.0, 'td_sell_price_std': 0.0,
                'td_vol': 0, 'td_price_std': 0.0
            }
        else:
            open, close = self.trade_buffer[0]['trade_price'], self.trade_buffer[-1]['trade_price']
            high, low = self.trade_buffer[0]['trade_price'], self.trade_buffer[0]['trade_price']
            td_buy_num, td_sell_num, td_buy_price_std, td_sell_price_std = 0, 0, 0, 0
            td_buy_price, td_sell_price,  td_buy_vol,  td_sell_vol = 0, 0, 0, 0
            # buy_price_sum, buy_price_sqsum = 0, 0
            # sell_price_sum, sell_price_sqsum = 0, 0
            price_sum, price_sqsum = 0, 0
            price_std = 0
            # 加入时间判断，保证在有新的成交数据输入到trade_buffer时，仍能正确地进行计算
            while self.trade_buffer and (datetime - self.trade_buffer[0]['datetime']).seconds >= 0:
                trade_info = self.trade_buffer.popleft()
                price = trade_info['trade_price']
                vol = trade_info['trade_volume']
                high = price if high < price else high
                low = price if low > price else low
                td_buy_num += (trade_info['side'] == 'B')
                td_sell_num += (trade_info['side'] == 'S')
                td_buy_price += ((trade_info['side'] == 'B') * price * vol)
                td_sell_price += ((trade_info['side'] == 'S') * price * vol)
                td_buy_vol += ((trade_info['side'] == 'B') * vol)
                td_sell_vol += ((trade_info['side'] == 'S') * vol)

                price_sum += price
                price_sqsum += price * price
            td_vol = td_buy_vol + td_sell_vol
            vwap = (td_buy_price + td_sell_price) / (td_buy_vol + td_sell_vol)

            if td_buy_vol > 0:
                td_buy_price = calculate_ratio(td_buy_price, td_buy_vol)
            else:
                td_buy_price = self.previous_trade_feat.get('td_buy_price', None)

            if td_sell_vol > 0:
                td_sell_price = calculate_ratio(td_sell_price, td_sell_vol)
            else:
                td_sell_price = self.previous_trade_feat.get('td_sell_price', None)

            td_num = (td_buy_num + td_sell_num)

            price_std = 0.0
            if td_num > 1:
                variance = (price_sqsum - (price_sum * price_sum) / td_num) / (td_num - 1)
                if variance < 0:
                    price_std = 0.0
                else:
                    price_std = math.sqrt(variance)

            features = {
                'datetime': datetime,
                'open': open,
                'close': close, #用快照数据获取close
                'high': high,
                'low': low,
                'td_buy_num': td_buy_num, 'td_sell_num': td_sell_num, 'td_buy_price': td_buy_price, 'td_sell_price': td_sell_price,
                'td_buy_vol': td_buy_vol, 'td_sell_vol': td_sell_vol,
                # 'td_buy_price_std': 0.0, 'td_sell_price_std': 0.0,
                'td_vol': td_vol, 'vwap': vwap, 'td_price_std': price_std
            }
        self.add_feature(features["datetime"], features)
        self.previous_trade_feat = features


    def build_entrust_features(self, datetime):
        ##########   构建特征所需函数   ######
        def calculate_ratio(numerator, denominator):
            try:
                return numerator / denominator
            except ZeroDivisionError:
                return 0

        ###################################

        if len(self.entrust_buffer) == 0:
            features = {
                'datetime': datetime,
                'en_buy_num': 0, 'en_sell_num': 0,
                'en_buy_price': self.previous_entrust_feat.get('en_buy_price', None),
                'en_sell_price': self.previous_entrust_feat.get('en_sell_price', None),
                'en_buy_vol': 0, 'en_sell_vol': 0, 'en_price_std': 0.0,
            }
        else:
            en_buy_num, en_sell_num = 0, 0
            en_buy_price, en_sell_price, en_buy_vol, en_sell_vol = 0, 0, 0, 0
            price_sum, price_sqsum = 0, 0
            en_price_std = 0
            # 加入时间判断，保证在有新的委托数据输入到entrust_buffer时，仍能正确地进行计算
            while self.entrust_buffer and (datetime - self.entrust_buffer[0]['datetime']).seconds >= 0:
                entrust_info = self.entrust_buffer.popleft()
                price = entrust_info['order_price']
                vol = entrust_info['order_volume']

                en_buy_num += (entrust_info['side'] == 'B')
                en_sell_num += (entrust_info['side'] == 'S')
                en_buy_price += ((entrust_info['side'] == 'B') * price * vol)
                en_sell_price += ((entrust_info['side'] == 'S') * price * vol)
                en_buy_vol += ((entrust_info['side'] == 'B') * vol)
                en_sell_vol += ((entrust_info['side'] == 'S') * vol)

                price_sum += price
                price_sqsum += price * price
            if en_buy_vol > 0:
                en_buy_price = calculate_ratio(en_buy_price, en_buy_vol)
            else:
                en_buy_price = None
            if en_sell_vol > 0:
                en_sell_price = calculate_ratio(en_sell_price, en_sell_vol)
            else:
                en_sell_price = None

            en_num = (en_buy_num + en_sell_num)
            if en_num > 1:
                variance = (price_sqsum - (price_sum * price_sum) / en_num) / (en_num - 1)
                en_price_std = math.sqrt(variance)

            features = {
                'datetime': datetime,
                'en_buy_num': en_buy_num, 'en_sell_num': en_sell_num, 'en_buy_price': en_buy_price, 'en_sell_price': en_sell_price,
                'en_buy_vol': en_buy_vol, 'en_sell_vol': en_sell_vol, 'en_price_std': en_price_std,
            }
        self.add_feature(features["datetime"], features)
        self.previous_entrust_feat = features

    def build_cancel_features(self, datetime):

        if len(self.cancel_buffer) == 0:
            features = {
                'datetime': datetime,
                'cancel_buy_num': 0, 'cancel_sell_num': 0, 'cancel_buy_vol': 0, 'cancel_sell_vol': 0,
                'cancel_buy_time_range': 0.0, 'cancel_sell_time_range': 0.0,
                'cancel_buy_time_med': 0.0, 'cancel_sell_time_med': 0.0,
            }
        else:
            cancel_buy_num, cancel_sell_num, cancel_buy_vol, cancel_sell_vol = 0, 0, 0, 0
            buy_time_list, sell_time_list = [], []
            # 加入时间判断，保证在有新的撤销数据输入到cancel_buffer时，仍能正确地进行计算
            while self.cancel_buffer and (datetime - self.cancel_buffer[0]['datetime']).seconds >= 0:
                cancel_info = self.cancel_buffer.popleft()
                vol = cancel_info['volume']
                time = cancel_info['datetime']
                try:
                    match_order = self.entrust_dict_by_appl_seq[cancel_info['appl_seq_num']]
                except KeyError as e:
                    log.error("当前撤单无法找到匹配的订单")
                    continue

                delta_time = (time - match_order['datetime']).total_seconds() * 1000
                if cancel_info['side'] == 'B':
                    cancel_buy_num += 1
                    cancel_buy_vol += vol
                    buy_time_list.append(delta_time)
                elif cancel_info['side'] == 'S':
                    cancel_sell_num += 1
                    cancel_sell_vol += vol
                    sell_time_list.append(delta_time)
                else:
                    log.error("Unknown side")

            if len(buy_time_list) > 0:
                cancel_buy_time_range = (np.max(buy_time_list) - np.min(buy_time_list)).item()
                cancel_buy_time_med = np.median(buy_time_list).item()
            else:
                cancel_buy_time_range = 0
                cancel_buy_time_med = 0
            if len(sell_time_list) > 0:
                cancel_sell_time_range = (np.max(sell_time_list) - np.min(sell_time_list)).item()
                cancel_sell_time_med = np.median(sell_time_list).item()
            else:
                cancel_sell_time_range = 0
                cancel_sell_time_med = 0
            features = {
                'datetime': datetime,
                'cancel_buy_num': cancel_buy_num, 'cancel_sell_num': cancel_sell_num, 'cancel_buy_vol': cancel_buy_vol, 'cancel_sell_vol': cancel_sell_vol,
                'cancel_buy_time_range': cancel_buy_time_range, 'cancel_sell_time_range': cancel_sell_time_range,
                'cancel_buy_time_med': cancel_buy_time_med, 'cancel_sell_time_med': cancel_sell_time_med,
            }
        self.add_feature(features["datetime"], features)
        self.previous_cancel_feat = features
