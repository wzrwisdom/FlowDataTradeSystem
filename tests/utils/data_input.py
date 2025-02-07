import pandas as pd
import sys
sys.path.insert(0, "../../")
from loguru import logger as log

from FlowDataTradeSystem.marketdata.data_adapter import CounterDataFetcher
from FlowDataTradeSystem.marketdata.counters.counterA import CounterAAdapter
from FlowDataTradeSystem.myenums.market_data_type import MarketDataType

def load_snap_info(data_fetcher, filepath, max_num=20):
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        if idx > max_num:
            break
        yield data_fetcher.fetch_data(MarketDataType.Snapshot, row)


def load_trade_info(data_fetcher, filepath, max_num=20):
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        if idx > max_num:
            break
        yield data_fetcher.fetch_data(MarketDataType.Transaction, row)

def load_order_info(data_fetcher, filepath, max_num=100):
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        if idx > max_num:
            break
        yield data_fetcher.fetch_data(MarketDataType.Entrust, row)


def load_all_data(data_fetcher):
    snap_filepath = 'data/snap.csv'
    trade_filepath = 'data/trade.csv'
    order_filepath = 'data/order.csv'

    snap_df = pd.read_csv(snap_filepath)
    trade_df = pd.read_csv(trade_filepath)
    order_df = pd.read_csv(order_filepath)

    snap_df = snap_df[snap_df.code == '510310.SH']
    trade_df = trade_df[trade_df.code == '510310.SH']
    order_df = order_df[order_df.code == '510310.SH']

    snap_df['tradetime'] = pd.to_datetime(snap_df['date']) + pd.to_timedelta(snap_df['time'])
    trade_df['tradetime'] = pd.to_datetime(trade_df['date']) + pd.to_timedelta(trade_df['time'])
    order_df['tradetime'] = pd.to_datetime(order_df['date']) + pd.to_timedelta(order_df['time'])

    condition = snap_df['tradetime'].dt.time >= pd.to_datetime('09:30:00').time()
    snap_df = snap_df[condition]
    condition = trade_df['tradetime'].dt.time >= pd.to_datetime('09:30:00').time()
    trade_df = trade_df[condition]
    # 由于撤单信息需要全量的逐笔委托信息，所以这儿不做时间筛选
    # condition = order_df['tradetime'].dt.time >= pd.to_datetime('09:25:00').time()
    # order_df = order_df[condition]
    # 创建迭代器
    trade_iter = iter(trade_df.iterrows())
    order_iter = iter(order_df.iterrows())
    snap_iter = iter(snap_df.iterrows())
    # 初始化当前交易和订单数据
    _, cur_trade = next(trade_iter)
    _, cur_order = next(order_iter)
    _, cur_snap = next(snap_iter)

    exclude_start = pd.Timestamp("11:30:00").time()
    exclude_end = pd.Timestamp("13:00:00").time()
    # 生成每隔3秒的时间序列
    all_times = pd.date_range(start="09:30:00", end="10:30:00", freq="3s").time
    # all_times = pd.date_range(start="09:30:00", end="14:57:00", freq="3S").time

    # 排除11:30到13:00的时间
    filtered_times = [t for t in all_times if t <= exclude_start or t >= exclude_end]
    last_snap = None
    for time in filtered_times:
        if time < pd.to_datetime('09:30:00').time():
            continue
        log.info(f"Processing snapshot at {time}")
        while cur_trade.tradetime.time() <= time or cur_order.tradetime.time() <= time:
            if cur_order.tradetime <= cur_trade.tradetime:
                yield data_fetcher.fetch_data(MarketDataType.Entrust, cur_order)
                _, cur_order = next(order_iter)
            else:
                yield data_fetcher.fetch_data(MarketDataType.Transaction, cur_trade)
                _, cur_trade = next(trade_iter)

        # 用上一个快照信息填补快照数据的空缺，频率为3s。
        if time >= cur_snap.tradetime.time() and time > (cur_snap.tradetime - pd.Timedelta('3s')).time():
            cur_snap.tradetime = cur_snap.tradetime.replace(hour=time.hour, minute=time.minute, second=time.second)
            cur_snap.time = str(time)
            yield data_fetcher.fetch_data(MarketDataType.Snapshot, cur_snap)
            last_snap = cur_snap
            _, cur_snap = next(snap_iter)
        else:
            if last_snap is not None:
                yield data_fetcher.fetch_data(MarketDataType.Snapshot, last_snap)
            else:
                print("There is no snap data!!!")

    # for _, snap_row in snap_df.iterrows():
    #     tradetime = snap_row.tradetime
    #
    #     if tradetime.time() < pd.to_datetime('09:30:00').time():
    #         continue
    #     print(f"Processing snapshot at {tradetime}")
    #     if tradetime.time() == pd.to_datetime('09:30:00').time():
    #         yield data_fetcher.fetch_data(MarketDataType.Snapshot, snap_row)
    #         continue
    #
    #     while cur_trade.tradetime < tradetime or cur_order.tradetime < tradetime:
    #         if cur_order.tradetime < cur_trade.tradetime:
    #             yield data_fetcher.fetch_data(MarketDataType.Entrust, cur_order)
    #             _, cur_order = next(order_iter)
    #         else:
    #             yield data_fetcher.fetch_data(MarketDataType.Transaction, cur_trade)
    #             _, cur_trade = next(trade_iter)
    #
    #     yield data_fetcher.fetch_data(MarketDataType.Snapshot, snap_row)


if __name__ == '__main__':
    counter = CounterAAdapter()
    data_fetcher = CounterDataFetcher(counter)
    for data in load_all_data(data_fetcher):
        print(data)