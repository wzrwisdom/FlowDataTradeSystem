from abc import ABC, abstractmethod
from order.order import Order
import uuid
from myenums.order_enum import *
from order.orderManager import OrderManager
from broker.brokerManager import BrokerManager


class Strategy(ABC):
    def __init__(self):
        self.order_manager = OrderManager()
        self.broker = BrokerManager()

    @abstractmethod
    def on_quote(self, context, data):
        """
        行情数据更新回调
        :param context:
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def on_entrust(self, context, data):
        """
        逐笔委托更新回调
        :param context:
        :param data:
        :return:
        """
        pass

    @abstractmethod
    def on_transaction(self, context, transaction):
        """
        逐笔成交更新回调
        :param context:
        :param transaction:
        :return:
        """
        pass

    @abstractmethod
    def on_order(self, context, order):
        """
        订单信息更新回调
        :param context:
        :param order:
        :return:
        """
        pass


    @abstractmethod
    def on_trade(self, context, trade):
        """
        订单成交回报回调
        :param context:
        :param trade:
        :return:
        """
        pass

    def generate_order_id(selfs):
        return str(uuid.uuid4())

    def buy(self, symbol, price, vol, close_flag=False):
        order_id = self.generate_order_id()
        if not close_flag:
            order = Order(order_id, symbol, price, vol, side=TradeDirectionEnum.Buy)
        else:
            order = Order(order_id, symbol, price, vol, side=TradeDirectionEnum.CloseSell)
        self.order_manager.send_order(order)
        self.broker.position_manager.update_position(order)
        return order_id

    def sell(self, symbol, price, vol, close_flag=False):
        order_id = self.generate_order_id()
        if not close_flag:
            order = Order(order_id, symbol, price, vol, side=TradeDirectionEnum.Sell)
        else:
            order = Order(order_id, symbol, price, vol, side=TradeDirectionEnum.CloseBuy)
        self.order_manager.send_order(order)
        self.broker.position_manager.update_position(order)
        return order_id



