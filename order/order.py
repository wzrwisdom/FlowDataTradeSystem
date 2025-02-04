class Order:
    def __init__(self, order_id, symbol, price, quantity, side, status='new'):
        self.order_id = order_id
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.side = side  # 'buy' or 'sell'
        self.status = status  # 'new', 'filled', 'cancelled', etc.

    def __repr__(self):
        return f"Order({self.order_id}, {self.symbol}, {self.side}, {self.price}, {self.quantity}, {self.status})"
