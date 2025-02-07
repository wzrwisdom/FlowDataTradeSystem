from FlowDataTradeSystem.myenums import *

class MarketData:
    '''
    :param data_type: 数据类型 "Snapshot", "Transaction", "Entrust"
    :param data: 数据类型：格式为字段
    '''
    def __init__(self, data_type: MarketDataType, data):
        self.data_type = data_type
        self.data = data

    def __repr__(self):
        return f"MarketData(type={self.data_type}, data={self.data})"

    def __getattr__(self, __name):
        if __name in self.data:
            return self.data[__name]
        else:
            raise AttributeError(f"'MarketData' object has no attribute '{__name}'")

    def __getitem__(self, key):
        # 支持通过下标访问字典
        return self.data[key]