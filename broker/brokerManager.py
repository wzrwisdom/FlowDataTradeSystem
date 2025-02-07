from FlowDataTradeSystem.broker.positionManager import PositionManager


class BrokerManager:
    def __init__(self):
        self.position_manager = PositionManager()

    def position(self, symbol):
        long_position = self.position_manager.get_position(symbol, "long")
        short_position = self.position_manager.get_position(symbol, "short")
        return long_position, short_position


    def calculate_pnl(self, symbol):
        return self.position_manager.get_realized_pnl(symbol)


    def get_portforlio_value(self):
        symbols = list(self.position_manager.history.keys())
        realized_pnl = 0
        for symbol in symbols:
            realized_pnl += self.calculate_pnl(symbol)
        return realized_pnl
