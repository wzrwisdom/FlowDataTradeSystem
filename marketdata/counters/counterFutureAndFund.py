import pandas as pd

from FlowDataTradeSystem.marketdata.data_adapter import DataAdapter

class CounterFutureAdapter(DataAdapter):

    def adapt_snapshot(self, raw_data):
        """适配快照数据"""
        return {
            'symbol': raw_data['symbol'],
            'datetime': raw_data['tradetime'],
            'last': raw_data['last']/10000,
            'high': raw_data['high']/10000,
            'low': raw_data['low']/10000,
            's1': raw_data['a1']/10000,
            'b1': raw_data['b1']/10000,
            'vol': raw_data['volume'],
            'bs_avg_price': (raw_data['a1'] + raw_data['b1']) / 2 / 10000,
            'sv1_sum': raw_data['a1_v'],
            'bv1_sum': raw_data['b1_v'],
            'total_turnover': raw_data['total_turnover']
            # 'vwap': raw_data['total_turnover']/(300 * raw_data['volume']),
        }


    def adapt_trade(self, raw_data):
        """适配成交数据"""
        pass

    def adapt_order(self, raw_data):
        pass