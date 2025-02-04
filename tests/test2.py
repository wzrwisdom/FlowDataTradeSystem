import datetime
import queue
import threading
import time

class Event:
    def __init__(self, type, data=None):
        self.type = type
        self.data = data

class MarketDataHandler:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def publish(self, data):
        for callback in self.subscribers:
            callback(Event("market_data", data))

class Strategy:
    def __init__(self, event_queue):
        self.event_queue = event_queue

    def on_market_data(self, event):
        # 简单的示例策略：如果最新价格高于 100，则发出买入信号
        if event.data["price"] > 100:
            order_event = Event("order", {"symbol": "TEST", "action": "BUY", "price": event.data["price"], "volume": 100})
            self.event_queue.put(order_event)
            print(f"发出买单：{order_event.data}")

class OrderExecutionHandler:
  def handle_order(self, event):
        print(f"执行订单：{event.data}")

def event_loop(event_queue, order_handler):
  while True:
      try:
          event = event_queue.get(timeout=0.1) #非阻塞式获取事件
          if event.type == "order":
              order_handler.handle_order(event)
      except queue.Empty:
          pass  # 没有事件时继续循环

if __name__ == "__main__":
    event_queue = queue.Queue()
    market_data_handler = MarketDataHandler()
    strategy = Strategy(event_queue)
    order_handler=OrderExecutionHandler()

    market_data_handler.subscribe(strategy.on_market_data)

    # 模拟行情数据
    def generate_market_data():
        while True:
            price = 90 + (datetime.datetime.now().microsecond % 20) #模拟价格在90-110之间波动
            data = {"symbol": "TEST", "price": price, "timestamp": datetime.datetime.now()}
            market_data_handler.publish(data)
            time.sleep(0.01)  # 模拟高频数据

    #启动事件循环和行情数据模拟
    event_loop_thread=threading.Thread(target=event_loop,args=(event_queue,order_handler))
    market_data_thread = threading.Thread(target=generate_market_data)

    event_loop_thread.start()
    market_data_thread.start()

    event_loop_thread.join()
    market_data_thread.join()