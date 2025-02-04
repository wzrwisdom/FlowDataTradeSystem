from marketdata.marketdata import MarketData
from myenums.market_data_type import MarketDataType
from abc import ABC, abstractmethod
from loguru import logger as log
class DataHandler(ABC):
    _registry = {}

    # def __init_subclass__(cls, **kwargs):
    #     super().__init_subclass__(**kwargs)
    #     DataHandler._registry[cls.__name__] = cls()

    def __init__(self):
        self.subscribers = []

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)

    @abstractmethod
    def publish(self, context, data):
        """遍历订阅器，发送信息"""
        pass

    @classmethod
    def publish_data(cls, context, data):
        if data.data_type == MarketDataType.Snapshot:
            cls._registry['SnapshotDataHandler'].publish(context, data)
        elif data.data_type == MarketDataType.Entrust:
            cls._registry['EntrustDataHandler'].publish(context, data)
        elif data.data_type == MarketDataType.Transaction:
            cls._registry['TransactionDataHandler'].publish(context, data)
        else:
            log.error(f'Unsupported data type: {data.data_type}')

class SnapshotDataHandler(DataHandler):
    def publish(self, context, data: MarketData):
        # assert data.data_type == MarketDataType.Snapshot, "data type must be Snapshot"

        for subscriber in self.subscribers:
            subscriber(context, data)

class EntrustDataHandler(DataHandler):
    def publish(self, context, data: MarketData):
        # assert data.data_type == MarketDataType.Entrust, "data type must be Entrust"
        for subscriber in self.subscribers:
            subscriber(context, data)

class TransactionDataHandler(DataHandler):
    def publish(self, context, data: MarketData):
        # assert data.data_type == MarketDataType.Transaction, "data type must be Transaction"
        for subscriber in self.subscribers:
            subscriber(context, data)