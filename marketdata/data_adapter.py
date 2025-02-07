from abc import ABC, abstractmethod

from FlowDataTradeSystem.myenums import *
from FlowDataTradeSystem.marketdata.marketdata import MarketData

class DataAdapter(ABC):

    @abstractmethod
    def adapt_snapshot(self, raw_data):
        """适配快照数据"""
        pass

    @abstractmethod
    def adapt_trade(self, raw_data):
        """适配成交数据"""
        pass

    @abstractmethod
    def adapt_order(self, raw_data):
        """适配委托数据"""
        pass


class CounterDataFetcher:
    def __init__(self, adapter: DataAdapter):
        """
        :param adapter: 适配器实例，根据柜台类型传入
        """
        self.adapter = adapter

    def fetch_data(self, data_type, raw_data):
        """
        :param data_type: 数据类型，例如 "snapshot", "trade", "order"
        :param raw_data: 从柜台获取的原始数据
        :return: 整理后的 MarketData 对象
        """
        if data_type == MarketDataType.Snapshot:
            data = self.adapter.adapt_snapshot(raw_data)
        elif data_type == MarketDataType.Transaction:
            data = self.adapter.adapt_trade(raw_data)
        elif data_type == MarketDataType.Entrust:
            data = self.adapter.adapt_order(raw_data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        return MarketData(data_type=data_type, data=data)
