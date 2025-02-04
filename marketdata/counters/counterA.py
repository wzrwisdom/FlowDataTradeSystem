import datetime
import pandas as pd

from marketdata.data_adapter import DataAdapter

class CounterAAdapter(DataAdapter):
    def adapt_snapshot(self, raw_data):
        # date = datetime.date.fromtimestamp(raw_data["date"])
        # time = (datetime.datetime.min + datetime.timedelta(milliseconds=raw_data["time"])).time()
        date = pd.to_datetime(raw_data['date'])
        time = pd.to_timedelta(raw_data['time'])
        bid_prices = []
        ask_prices = []
        bid_volumes = []
        ask_volumes = []
        for i in range(1, 10+1):
            bid_prices.append(raw_data[f'bid{i}'])
            ask_prices.append(raw_data[f'ask{i}'])
            bid_volumes.append(raw_data[f'bid_size{i}'])
            ask_volumes.append(raw_data[f'ask_size{i}'])
        return {
            "symbol": raw_data["code"],
            "datetime": date + time,
            "last_price": raw_data["Last"],
            # "time": time,
            "bid_prices": bid_prices,
            "ask_prices": ask_prices,
            "bid_volumes": bid_volumes,
            "ask_volumes": ask_volumes,
        }

    def adapt_trade(self, raw_data):
        # date = datetime.date.fromtimestamp(raw_data["date"])
        # time = (datetime.datetime.min + datetime.timedelta(milliseconds=raw_data["time"])).time()
        date = pd.to_datetime(raw_data['date'])
        time = pd.to_timedelta(raw_data['time'])
        return {
            "symbol": raw_data["code"],
            "datetime": date + time,
            # "date": date,
            # "time": time,
            "appl_seq_num": raw_data["index"], # 频道编号
            "bid_appl_seq_num": raw_data["buy_index"], # 买方委托索引
            "ask_appl_seq_num": raw_data["sell_index"], # 卖方委托索引
            "side": raw_data["bs_flag"], # 买卖方向
            "trade_price": raw_data["trade_price"], # 成交价格
            "trade_volume": raw_data["trade_volume"], # 成交量
        }

    def adapt_order(self, raw_data):
        # date = datetime.date.fromtimestamp(raw_data["date"])
        # time = (datetime.datetime.min + datetime.timedelta(milliseconds=raw_data["time"])).time()
        date = pd.to_datetime(raw_data['date'])
        time = pd.to_timedelta(raw_data['time'])
        return {
            "symbol": raw_data["code"],
            "datetime": date + time,
            # "date": date,
            # "time": time,
            "appl_seq_num": raw_data["order"], # 频道编号
            "biz_index": raw_data["index"], # 业务序列
            "order_type": raw_data["order_kind"], # 订单类型
            "side": raw_data["function_code"], # 买卖方向
            "order_price": raw_data["price"],
            "order_volume": raw_data["volume"],
        }
