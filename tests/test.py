import threading
import queue
import time
from typing import Callable

# Market Data Module
class MarketData:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback: Callable):
        self.subscribers.append(callback)

    def feed_data(self, data):
        for callback in self.subscribers:
            callback(data)

# Trading Module
class TradingEngine:
    def __init__(self):
        self.order_queue = queue.Queue()

    def place_order(self, order):
        self.order_queue.put(order)
        print(f"Order placed: {order}")

    def process_orders(self):
        while True:
            if not self.order_queue.empty():
                order = self.order_queue.get()
                self.execute_order(order)

    def execute_order(self, order):
        print(f"Order executed: {order}")

# Risk Management Module
class RiskManager:
    def __init__(self):
        self.max_position = 100
        self.current_position = 0

    def check_risk(self, order):
        if self.current_position + order['quantity'] > self.max_position:
            print("Risk limit exceeded! Order rejected.")
            return False
        return True

# Strategy Module
class Strategy:
    def __init__(self, market_data: MarketData, trading_engine: TradingEngine, risk_manager: RiskManager):
        self.market_data = market_data
        self.trading_engine = trading_engine
        self.risk_manager = risk_manager
        self.market_data.subscribe(self.on_market_data)

    def on_market_data(self, data):
        print(f"Received market data: {data}")
        # Example strategy: Place an order if price is below a threshold
        if data['price'] < 50:
            order = {'symbol': data['symbol'], 'price': data['price'], 'quantity': 10}
            if self.risk_manager.check_risk(order):
                self.trading_engine.place_order(order)

# Logger Module
class Logger:
    @staticmethod
    def log(message):
        print(f"[LOG] {message}")

# Main Framework
if __name__ == "__main__":
    # Initialize components
    market_data = MarketData()
    trading_engine = TradingEngine()
    risk_manager = RiskManager()
    strategy = Strategy(market_data, trading_engine, risk_manager)

    # Start trading engine in a separate thread
    trading_thread = threading.Thread(target=trading_engine.process_orders, daemon=True)
    trading_thread.start()

    # Simulate market data feed
    for i in range(5):
        simulated_data = {'symbol': 'AAPL', 'price': 45 + i * 2}  # Prices: 45, 47, 49, ...
        market_data.feed_data(simulated_data)
        time.sleep(1)
