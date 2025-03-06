from collections import defaultdict
from loguru import logger as log
from FlowDataTradeSystem.myenums.order_enum import TradeDirectionEnum


class PositionManager:
    def __init__(self):
        # 存储每个品种的持仓量
        self.short_positions = defaultdict(int)  # symbol -> 当前空仓持仓量
        self.long_positions = defaultdict(int)  # symbol -> 当前多仓持仓量
        # 存储每个品种的历史买入价和成交量，计算已实现盈亏
        self.history = defaultdict(list)  # symbol -> [(买入价/卖出价, 数量, 买卖标志)]

    def update_position(self, order):
        """
        更新持仓：买入时增加，卖出时减少
        :param symbol: 交易品种
        :param quantity: 交易数量
        :param price: 交易价格
        :param side: 'buy' 或 'sell'
        """
        side = order.side
        symbol = order.symbol
        quantity = order.quantity
        price = order.price
        if side == TradeDirectionEnum.Buy:
            # 记录买入操作
            self.long_positions[symbol] += quantity
            self.history[symbol].append((price, quantity, 'Buy'))  # 将买入价和数量存储
        elif side == TradeDirectionEnum.CloseBuy:
            # 记录卖出操作
            if self.long_positions[symbol] >= quantity:
                self.long_positions[symbol] -= quantity
                self.history[symbol].append((price, quantity, 'Sell'))
            else:
                log.warning(f"Not enough {symbol} to sell!")
        elif side == TradeDirectionEnum.Sell:
            # 记录买入操作
            self.short_positions[symbol] += quantity
            self.history[symbol].append((price, quantity, 'Sell'))  # 将买入价和数量存储
        elif side == TradeDirectionEnum.CloseSell:
            # 记录卖出操作
            if self.short_positions[symbol] >= quantity:
                self.short_positions[symbol] -= quantity
                self.history[symbol].append((price, quantity, 'Buy'))
            else:
                log.warning(f"Not enough {symbol} to Buy!")
        else:
            log.error("Invalid side. Should be 'buy' or 'sell'.")

    def get_position(self, symbol, side):
        """获取某个品种的当前多头或空头仓位"""
        if side == 'long':
            return self.long_positions.get(symbol, 0)
        elif side == 'short':
            return self.short_positions.get(symbol, 0)

    def get_realized_pnl(self, symbol):
        """计算已实现盈亏：卖出操作时，基于买入价格和卖出价格"""
        realized_pnl = 0
        buy_amount = 0
        sell_amount = 0
        buy_vol = 0
        sell_vol = 0
        for price, qty, bs_flag in self.history[symbol]:
            if bs_flag == 'Buy':
                buy_amount += (qty*price)
                buy_vol += qty
            elif bs_flag == 'Sell':
                sell_amount += (qty*price)
                sell_vol += qty
        assert buy_vol == sell_vol
        realized_pnl = sell_amount - buy_amount
        return realized_pnl

