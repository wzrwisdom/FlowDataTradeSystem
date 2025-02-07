import pandas as pd

def prepareQuote(data, time):
    new_data = {}
    new_data['datetime'] = pd.to_datetime(time)
    new_data['symbol'] = data.instrument_id + ".SH" if data.exchange_id == "SSE" else ".SZ"
    new_data['ask_prices'] = data.ask_price
    new_data['ask_volumes'] = data.ask_volume
    new_data['bid_prices'] = data.bid_price
    new_data['bid_volumes'] = data.bid_volume
    new_data['last_price'] = data.last_price
    return new_data

def prepareOrder(data):
    new_data = {}
    new_data['datetime'] = pd.to_datetime(data.insert_time) + pd.Timedelta(hours=8)
    new_data['symbol'] = data.instrument_id + ".SH" if data.exchange_id == "SSE" else ".SZ"
    new_data['appl_seq_num'] = data.seq
    new_data['biz_index'] = data.biz_index
    new_data['order_type'] = 'D' if data.price_type == "Cancel" else 'O'
    new_data['side'] = 'B' if data.side == 'Buy' else 'S'
    new_data['order_price'] = data.price
    new_data['order_volume'] = data.volume
    return new_data


def prepareTrade(data):
    new_data = {}
    new_data['datetime'] = pd.to_datetime(data.insert_time) + pd.Timedelta(hours=8)
    new_data['symbol'] = data.instrument_id + ".SH" if data.exchange_id == "SSE" else ".SZ"
    new_data['appl_seq_num'] = data.seq
    new_data['bid_appl_seq_num'] = data.bid_no
    new_data['ask_appl_seq_num'] = data.ask_no
    new_data['side'] = 'B' if data.side == 'Buy' else 'S'
    new_data['trade_price'] = data.trade_price
    new_data['trade_volume'] = data.trade_volume
    return new_data
