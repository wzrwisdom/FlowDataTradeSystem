from collections import deque
import math
from loguru import logger as log
import numpy as np
import pandas as pd

class FutureFeatureBuilder:
    def __init__(self):
        self.maxlen = 1000
        self.history_feat_dict = {}
        self.history_time = deque(maxlen=self.maxlen)
        self.perivous_snap_feat = {}


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

    def build_snap_features(self, data):
        features = {}
        vwap = data['total_turnover'] / (300 * data['volume'])

        features.update({
            'datetime': data['datetime'],
            'last': data['last'],
            'vol': data['volume'],
            's1': data['s1'],
            'b1': data['b1'],
            'sv1_sum': data['sv1_sum'],
            'bv1_sum': data['bv1_sum'],
            'bs_avg_price': data['bs_avg_price'],
            'vwap': vwap,
        })

        self.history_time.append(features['datetime'])
        self.add_feature(features['datetime'], features)
        self.previous_snap_feat = features
