import time

class OrderManager:
    def __init__(self):
        self.orders = {}  # Order ID -> Order object

    def send_order(self, order):
        # 将订单添加到订单列表
        self.orders[order.order_id] = order
        print(f"Placed order: {order}")
        # 应该通过接口向交易所提交订单
        self._send_order_to_exchange(order)

    def cancel_order(self, order_id):
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = 'cancelled'
            print(f"Cancelled order: {order}")
            self._cancel_order_from_exchange(order)
        else:
            print(f"Order with ID {order_id} not found!")

    def get_order_status(self, order_id):
        return self.orders.get(order_id, None)

    def _send_order_to_exchange(self, order):
        # 模拟发送订单到交易所
        # time.sleep(0.01)  # 模拟延迟
        order.status = 'filled'  # 假设订单填充成功
        print(f"Order {order.order_id} filled!")

    def _cancel_order_from_exchange(self, order):
        # 模拟撤销订单
        # time.sleep(0.01)  # 模拟延迟
        print(f"Order {order.order_id} cancelled on exchange.")