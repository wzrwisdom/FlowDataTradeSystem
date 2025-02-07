
import numpy as np
from FlowDataTradeSystem.factor.factors.base import Factor
from FlowDataTradeSystem.factor.factor_func import *
import pandas as pd
from sortedcontainers import SortedList
import talib as ta
import warnings
warnings.simplefilter("error", RuntimeWarning)

class CloseFactor(Factor):
    name = "close_ret"

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        close = builder.get_recent_features(name="close", window=shift + 1)
        if len(close) < shift + 1:
            return None
        res = close.iloc[-1] / close.iloc[-shift-1] - 1
        return res.item() if not np.isnan(res.item()) else None


class ReturnVolProductFactor(Factor):
    name = "ret_v_prod"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        close = builder.get_recent_features(name="close", window=window + 1)
        vol = builder.get_recent_features(name="td_vol", window=window)
        if len(close) < int(window/3) + 1:
            return None

        previous_list = self.state.get("previous_list", None)
        if previous_list is None:
            previous_list = []
            # 计算收益率
            values = [
                (close.iloc[i + 1] - close.iloc[i]) / close.iloc[i] * vol.iloc[i+1]
                for i in range(len(close) - 1)
            ]
            for value in values:
                previous_list.append(value)
            self.state["previous_list"] = previous_list

            ranked = ts_rank(pd.Series(values), window)
            return ranked.iloc[-1].item()
        else:
            # 插入新价格
            new_val = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-1] * vol.iloc[-1]
            previous_list.append(new_val)

            # 如果超过10个价格，移除最早的一个
            if len(previous_list) > window:
                previous_list.pop(0)  # 删除最小的，保证维护10个价格

            # 获取排名 (归一化到 [0, 1])
            ranked = ts_rank(pd.Series(previous_list), window)
            return ranked.iloc[-1].item()


class WB1Factor(Factor):
    name = "wb1_tsrank"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        wb1 = builder.get_recent_features(name="wb1", window=window)
        if len(wb1) < int(window/3):
            return None
        ranked = ts_rank(wb1, window)
        return ranked.iloc[-1].item()


class WB10Factor(Factor):
    name = "wb10_tsrank"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        wb10 = builder.get_recent_features(name="wb10", window=window)
        if len(wb10) < int(window/3):
            return None
        ranked = ts_rank(wb10, window)
        return ranked.iloc[-1].item()


class SpreadFactor(Factor):
    name = "spread1"

    def compute(self, builder, **kwargs):
        s1 = builder.get_recent_features(name="s1", window=1)
        b1 = builder.get_recent_features(name="b1", window=1)
        res = s1[0] - b1[0]
        return res.item()


class EnSellSumPriceFactor(Factor):
    name = 'en_s_sumprice_tsrank'

    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window_tot = window1 + window2 - 1
        en_sell_p = builder.get_recent_features(name="en_sell_price", window=window_tot)
        en_sell_v = builder.get_recent_features(name="en_sell_vol", window=window_tot)
        avg_p = builder.get_recent_features(name="bs_avg_price", window=window1)

        if len(en_sell_p) < int((window1+window2)/3)-1:
            return None

        sell_avg_p = avg_price(en_sell_p, en_sell_v, window2)[-window1:]
        sell_avg_p.reset_index(inplace=True, drop=True)
        res = ts_rank(sell_avg_p - avg_p, window1)
        return res.iloc[-1].item()



class EnBuySumPriceFactor(Factor):
    name = 'en_b_sumprice_tsrank'

    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window_tot = window1 + window2 - 1
        en_buy_p = builder.get_recent_features(name="en_buy_price", window=window_tot)
        en_buy_v = builder.get_recent_features(name="en_buy_vol", window=window_tot)
        avg_p = builder.get_recent_features(name="bs_avg_price", window=window1)

        if len(en_buy_p) < int((window1+window2)/3)-1:
            return None

        buy_avg_p = avg_price(en_buy_p, en_buy_v, window2)[-window1:]
        buy_avg_p.reset_index(inplace=True, drop=True)
        res = ts_rank(buy_avg_p - avg_p, window1)
        return res.iloc[-1].item()


class OBPriceSpreadFactor(Factor):
    name = 'OB_price_spread_tsrank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        s1 = builder.get_recent_features(name="s1", window=window)
        b1 = builder.get_recent_features(name="b1", window=window)
        if len(s1) < int(window/3):
            return None
        res = ts_rank(np.round(s1 - b1,3), window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class BSv5Factor(Factor):
    name = 'bs_v5_tsrank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        bv5_sum = builder.get_recent_features(name="bv5_sum", window=window)
        sv5_sum = builder.get_recent_features(name="sv5_sum", window=window)
        if len(bv5_sum) < int(window/3):
            return None
        res = ts_rank(bv5_sum - sv5_sum, window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class DeridBid1Factor(Factor):
    name = 'DeriPBid1'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        b1 = builder.get_recent_features(name="b1", window=shift + 1)
        if len(b1) < shift + 1:
            return None

        res = delta(b1, shift)
        return res.iloc[-1].item()


class EnSellPriceFactor(Factor):
    name = "en_s_price_tsrank"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        s1 = builder.get_recent_features(name="s1", window=window)
        en_sell_p = builder.get_recent_features(name="en_sell_price", window=window)
        avg_p = builder.get_recent_features(name="bs_avg_price", window=window)

        if len(en_sell_p) < int(window/3):
            return None
        en_sell_p = en_sell_p.round(5)
        s1 = s1.round(4)
        avg_p = avg_p.round(4)
        res = ts_rank(fill_na_v2(en_sell_p, s1) - avg_p, window=window)
        return res.iloc[-1].item()


class DEMAFactor(Factor):
    name = "DEMA"

    def compute(self, builder, **kwargs):
        timeperiod = self.kwargs["timeperiod"]
        shift = self.kwargs["shift"]
        window_tot = 2 * timeperiod + shift - 1
        close = builder.get_recent_features(name="close", window=None)

        if len(close) < window_tot:
            return None

        close = close.round(4)
        res = ta.DEMA(close, timeperiod)
        res = ret(res, shift)
        return res.iloc[-1].item()


class BestVImbalanceFactor(Factor):
    name = 'best_v_imbalance_tsrank'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        window = self.kwargs["window"]
        window_tot = window + shift
        bv = builder.get_recent_features(name="bv1_sum", window=window_tot)
        sv = builder.get_recent_features(name="sv1_sum", window=window_tot)
        b1 = builder.get_recent_features(name="b1", window=window_tot)
        s1 = builder.get_recent_features(name="s1", window=window_tot)

        if len(b1) < int(window/3) + shift:
            return None

        b_flag1 = b1 == b1.shift(shift)
        s_flag1 = s1 == s1.shift(shift)
        b_flag2 = b1 < b1.shift(shift)
        s_flag2 = s1 > s1.shift(shift)

        bv_change = np.where(b_flag2, 0, np.where(b_flag1, bv - bv.shift(shift), bv))
        sv_change = np.where(s_flag2, 0, np.where(s_flag1, sv - sv.shift(shift), sv))
        res = bv_change - sv_change

        res = ts_rank(pd.Series(res), window)
        return res.iloc[-1].item()
#
#
# class LeadingSpanBFactor(Factor):
#     name = 'LeadingSpanB'
#
#     def compute(self, builder, **kwargs):
#         window = self.kwargs["window"]
#         shift = self.kwargs["shift"]
#         window_tot = shift + window
#         high = builder.get_recent_features(name="high", window=window_tot)
#         low = builder.get_recent_features(name="low", window=window_tot)
#         if len(low) < window:
#             return None
#
#         ts_high = ts_max(high, window)
#         ts_low = ts_min(low, window)
#         timeperiod = window * 52
#         ts_high = ts_mean(ts_high, timeperiod)
#         ts_low = ts_mean(ts_low, timeperiod)
#         res = (ts_high + ts_low) / 2.0
#         res = delta(res, shift) / res
#         return res.iloc[-1].item() if np.isnan(res.iloc[-1]) else None
# #
#
# class BaseLineFactor(Factor):
#     name = 'BaseLine'
#
#     def compute(self, builder, **kwargs):
#         window = self.kwargs["window"]
#         shift = self.kwargs["shift"]
#         window_tot = shift + window
#         high = builder.get_recent_features(name="high", window=window_tot)
#         low = builder.get_recent_features(name="low", window=window_tot)
#         if len(low) < window:
#             return None
#
#         ts_high = ts_max(high, window)
#         ts_low = ts_min(low, window)
#         timeperiod = window * 26
#         ts_high = ts_mean(ts_high, timeperiod)
#         ts_low = ts_mean(ts_low, timeperiod)
#         res = (ts_high + ts_low) / 2.0
#         res = delta(res, shift) / res
#         return res.iloc[-1].item()


class EnBSPriceDiffFactor(Factor):
    name = 'en_bs_price_diff'

    def compute(self, builder, **kwargs):
        en_buy_p = builder.get_recent_features(name="en_buy_price", window=1)
        en_sell_p = builder.get_recent_features(name="en_sell_price", window=1)
        b1 = builder.get_recent_features(name="b1", window=1)
        s1 = builder.get_recent_features(name="s1", window=1)
        avg_p = builder.get_recent_features(name="bs_avg_price", window=1)

        res = (fill_na_v2(en_buy_p, b1) - fill_na_v2(en_sell_p, s1)) / avg_p
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class EnSellRet10Factor(Factor):
    name = 'en_s_ret10_tsrank'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        window = self.kwargs["window"]
        window_tot = shift + window
        ssv10_sum = builder.get_recent_features(name="ssv10_sum", window=window_tot)
        sv10_sum = builder.get_recent_features(name="sv10_sum", window=window_tot)
        if len(ssv10_sum) < int(window/3)+shift:
            return None
        res = ts_rank(ret(ssv10_sum / sv10_sum, shift), window=window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class EnBuyRet10Factor(Factor):
    name = 'en_b_ret10_tsrank'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        window = self.kwargs["window"]
        window_tot = shift + window
        bbv10_sum = builder.get_recent_features(name="bbv10_sum", window=window_tot)
        bv10_sum = builder.get_recent_features(name="bv10_sum", window=window_tot)

        if len(bbv10_sum) < int(window/3)+shift:
            return None

        res = ts_rank(ret(bbv10_sum / bv10_sum, shift), window=window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class EnSellP10Factor(Factor):
    name = 'en_s_p10_tsrank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        ssv10_sum = builder.get_recent_features(name="ssv10_sum", window=window)
        sv10_sum = builder.get_recent_features(name="sv10_sum", window=window)
        if len(ssv10_sum) < int(window/3):
            return None
        res = ts_rank(ssv10_sum / sv10_sum, window=window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class CanEnVolRatioFactor(Factor):
    name = 'can_en_v_ratio'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        en_sell_v = builder.get_recent_features(name="en_sell_vol", window=window)
        en_buy_v = builder.get_recent_features(name="en_buy_vol", window=window)
        can_sell_v = builder.get_recent_features(name="cancel_sell_vol", window=window)
        can_buy_v = builder.get_recent_features(name="cancel_buy_vol", window=window)

        if len(en_sell_v) < int(window/3):
            return 0

        buy_power = ts_sum(en_buy_v, window) + ts_sum(can_sell_v, window)
        sell_power = ts_sum(en_sell_v, window) + ts_sum(can_buy_v, window)

        res = (buy_power - sell_power) / (buy_power + sell_power)
        res.fillna(0, inplace=True)
        return res.iloc[-1].item()


class OBPriceDeriFactor(Factor):
    name = 'OB_price_2derivative_tsrank'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        window = self.kwargs["window"]
        window_tot = 3*shift + window
        avg_p = builder.get_recent_features(name="bs_avg_price", window=window_tot)
        if len(avg_p) < int(window/3)+shift:
            return None

        avg_p = avg_p.round(4)
        avg_p_shift = move(avg_p, shift)
        avg_p_twoshift = move(avg_p, shift * 2)

        res = (avg_p_twoshift + avg_p - 2 * avg_p_shift) / avg_p
        res = ts_rank(res, window=window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class CanEnVolRatioTRankFactor(Factor):
    name = 'can_en_v_ratio_tsrank'

    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window = window1 + window2 - 1
        en_sell_v = builder.get_recent_features(name="en_sell_vol", window=window)
        en_buy_v = builder.get_recent_features(name="en_buy_vol", window=window)
        can_sell_v = builder.get_recent_features(name="cancel_sell_vol", window=window)
        can_buy_v = builder.get_recent_features(name="cancel_buy_vol", window=window)

        if len(en_sell_v) < int(window1/3) :
            return None

        buy_power = ts_sum(en_buy_v, window2) + ts_sum(can_sell_v, window2)
        sell_power = ts_sum(en_sell_v, window2) + ts_sum(can_buy_v, window2)

        res = (buy_power - sell_power) / (buy_power + sell_power)
        res.fillna(0, inplace=True)
        res = ts_rank(res, window1)
        return res.iloc[-1].item()


class EnBuyPriceFactor(Factor):
    name = 'en_b_price_tsrank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        en_buy_p = builder.get_recent_features(name="en_buy_price", window=window)
        b1 = builder.get_recent_features(name="b1", window=window)
        avg_p = builder.get_recent_features(name="bs_avg_price", window=window)

        if len(en_buy_p) < int(window/3):
            return None

        res = ts_rank(fill_na_v2(en_buy_p, b1) - avg_p, window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class EnBuyP10Factor(Factor):
    name = 'en_b_p10_tsrank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        bbv10_sum = builder.get_recent_features(name="bbv10_sum", window=window)
        bv10_sum = builder.get_recent_features(name="bv10_sum", window=window)

        if len(bbv10_sum) < int(window/3):
            return None


        res = ts_rank(bbv10_sum / bv10_sum, window=window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class RetNCorrFactor(Factor):
    name = "ret_n_corr"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        close = builder.get_recent_features(name="close", window=window+1)
        td_buy_n = builder.get_recent_features(name="td_buy_num", window=window)
        td_sell_n = builder.get_recent_features(name="td_sell_num", window=window)

        if len(close) < int(window/2)+1:
            return None
        size = len(td_buy_n)
        ret_v = ret(close)[-size:].reset_index(drop=True)
        res = ts_corr(ret_v, (td_buy_n + td_sell_n), window=window)
        if np.isnan(res.iloc[-1]):
            return None
        return res.iloc[-1].item()


class TSIFactor(Factor):
    name = 'TSI'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        timeperiod1 = shift * 25
        timeperiod2 = shift * 13
        window_tot = timeperiod1 + shift + timeperiod2
        close = builder.get_recent_features(name="close", window=None)
        if len(close) < window_tot:
            return None

        det_c = delta(close, shift)
        ema1 = ta.EMA(det_c, timeperiod1)
        ema2 = ta.EMA(det_c.abs(), timeperiod1)
        res = ta.EMA(ema1, timeperiod2) * 100 / ta.EMA(ema2, timeperiod2)
        return res.iloc[-1].item()


# class ConversionLineFactor(Factor):
#     name = 'ConversionLine'
#
#     def compute(self, builder, **kwargs):
#         window = self.kwargs["window"]
#         shift = self.kwargs["shift"]
#         window_tot = shift + window
#         high = builder.get_recent_features(name="high", window=window_tot)
#         low = builder.get_recent_features(name="low", window=window_tot)
#         if len(low) < window:
#             return None
#
#         ts_high = ts_max(high, window)
#         ts_low = ts_min(low, window)
#         timeperiod = window * 9
#         ts_high = ts_mean(ts_high, timeperiod)
#         ts_low = ts_mean(ts_low, timeperiod)
#         res = (ts_high + ts_low) / 2.0
#         res = delta(res, shift) / res
#         return res.iloc[-1].item()


class TradeVolRatioFactor(Factor):
    name = "td_v_ratio"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_buy_v = builder.get_recent_features(name="td_buy_vol", window=window)
        td_sell_v = builder.get_recent_features(name="td_sell_vol", window=window)
        if len(td_buy_v) < int(window/3):
            return None

        res = fill_na((td_buy_v - td_sell_v) / (td_buy_v + td_sell_v))
        res = ts_rank(res, window)
        return res.iloc[-1].item()


class EnSellRet5Factor(Factor):
    name = 'en_s_ret5_tsrank'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        window = self.kwargs["window"]
        window_tot = shift + window
        ssv5_sum = builder.get_recent_features(name="ssv5_sum", window=window_tot)
        sv5_sum = builder.get_recent_features(name="sv5_sum", window=window_tot)
        if len(ssv5_sum) < int(window/3) + shift:
            return None


        res = ts_rank(ret(ssv5_sum / sv5_sum, shift), window=window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class TradeBuySellVolRatioFactor(Factor):
    name = 'td_buy_and_sell_vol_rela_ratio'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_buy_v = builder.get_recent_features(name="td_buy_vol", window=window)
        td_sell_v = builder.get_recent_features(name="td_sell_vol", window=window)
        if len(td_buy_v) < int(window/3):
            return None

        buy_v = ts_sum(td_buy_v, window)
        sell_v = ts_sum(td_sell_v, window)
        res = (buy_v - sell_v) / (buy_v + sell_v)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class EnBuySellVolRatioFactor(Factor):
    name = "en_buy_and_sell_vol_rela_ratio"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        en_buy_v = builder.get_recent_features(name="en_buy_vol", window=window)
        en_sell_v = builder.get_recent_features(name="en_sell_vol", window=window)
        if len(en_buy_v) < int(window/3):
            return None

        buy_v = ts_sum(en_buy_v, window)
        sell_v = ts_sum(en_sell_v, window)
        res = (buy_v - sell_v) / (buy_v + sell_v)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class OHLCRatioFactor(Factor):
    name = "ohlc_rat"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        o = builder.get_recent_features(name="open", window=window)
        c = builder.get_recent_features(name="close", window=window)
        h = builder.get_recent_features(name="high", window=window)
        l = builder.get_recent_features(name="low", window=window)

        if len(o) < min(100, int(window/2)):
            return None
        o_r = o.rolling(window=window, min_periods=min(100, int(window / 2))).apply(lambda x: x[~np.isnan(x)][0],
                                                                                    raw=True)
        h_r = h.rolling(window=window, min_periods=min(100, int(window / 2))).max()
        l_r = l.rolling(window=window, min_periods=min(100, int(window / 2))).min()
        c_r = c.rolling(window=window, min_periods=min(100, int(window / 2))).apply(lambda x: x[~np.isnan(x)][-1],
                                                                                    raw=True)
        res = (c_r - o_r) / (h_r - l_r)
        res.where((h_r - l_r) != 0, 0, inplace=True)
        return res.iloc[-1].item()


class TradePVRatioFactor(Factor):
    name = 'td_p_v_ratio'

    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window = window1 + window2
        vwap = builder.get_recent_features(name="vwap", window=window)
        vol = builder.get_recent_features(name="td_vol", window=window-1)
        if len(vwap) < int(window1/3):
            return None
        delta_vwap = delta(vwap, window2)[-window1:].reset_index(drop=True)
        delta_vwap = delta_vwap.round(10)
        vol_sum = ts_sum(vol, window2)[-window1:].reset_index(drop=True)
        res = (delta_vwap / vol_sum)
        res.fillna(0, inplace=True)

        res = ts_rank(res, window1)
        return res.iloc[-1].item()


class TradeVolInHihgPriceFactor(Factor):
    name = 'trade_vol_in_high_price'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        perc = self.kwargs["perc"]
        window_tot = 2 * window - 1
        td_buy_v = builder.get_recent_features(name="td_buy_vol", window=window)
        td_sell_v = builder.get_recent_features(name="td_sell_vol", window=window)
        price = builder.get_recent_features(name="close", window=window_tot)
        if len(price) < window:
            return 0

        # size = len(price) - window + 1
        # vol = (td_buy_v + td_sell_v)[-size:].reset_index(drop=True)
        vol = (td_buy_v + td_sell_v)
        prank = price.rolling(window=window).rank(pct=True)[-window:].reset_index(drop=True)
        n_in_region = vol[prank > perc + 1e-7].sum()
        n_total = vol.sum()
        try:
            res = n_in_region / n_total
        except RuntimeWarning as e:
            return 0
        return res.item()

class HCPFactor(Factor):
    name = 'HCP'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        close = builder.get_recent_features(name="close", window=window)
        td_buy_p = builder.get_recent_features(name="td_buy_price", window=window)

        if len(close) < window:
            return None

        td_buy_p.ffill(inplace=True)
        new_p = td_buy_p.where(td_buy_p > close.iloc[-1] + 1e-5, 0)
        res = np.nanmean(new_p, axis=0) / close.iloc[-1]
        return res.item()

class CloseAdjustFactor(Factor):
    name = 'close_adjusted'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]

        close = builder.get_recent_features(name="close", window=window)
        if len(close) < int(window/3):
            return None

        res = (close - ts_mean(close, window)) / ts_std(close, window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class LCPFactor(Factor):
    name = 'LCP'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        close = builder.get_recent_features(name="close", window=window)
        td_sell_p = builder.get_recent_features(name="td_sell_price", window=window)

        if len(close) < window:
            return None

        td_sell_p.ffill(inplace=True)
        new_p = td_sell_p.where(td_sell_p < close.iloc[-1] - 1e-5, 0)
        res = np.nanmean(new_p, axis=0) / close.iloc[-1]
        return res.item()


class TradeEnVolRatioBuyDirFactor(Factor):
    name = "td_and_en_vol_ratio_buy_dir"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_buy_v = builder.get_recent_features(name="td_buy_vol", window=window)
        en_buy_v = builder.get_recent_features(name="en_buy_vol", window=window)
        if len(td_buy_v) < int(window/3):
            return 0

        res = ts_sum(td_buy_v, window)/ts_sum(en_buy_v, window)
        res = fill_na(res)
        return res.iloc[-1].item()


class TradeBuyFactor(Factor):
    name = 'td_buy_rank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_buy_p = builder.get_recent_features(name="td_buy_price", window=window)
        td_buy_v = builder.get_recent_features(name="td_buy_vol", window=window)

        if len(td_buy_p) < int(window/3):
            return None

        res = td_buy_p * td_buy_v
        res = ts_rank(res, window)
        return res.iloc[-1].item()


class TradePowerFactor(Factor):
    name = 'bs_td_power_rough'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        buy_v = builder.get_recent_features(name="td_buy_vol", window=window)
        buy_p = builder.get_recent_features(name="td_buy_price", window=window)
        sell_v = builder.get_recent_features(name="td_sell_vol", window=window)
        sell_p = builder.get_recent_features(name="td_sell_price", window=window)
        close = builder.get_recent_features(name="close", window=1)

        if len(buy_v) < window:
            return None

        close = close.item()
        buy_power = buy_v * buy_p / close
        sell_power = sell_v * ((2*close - sell_p) / close)
        buy_power = np.nansum(buy_power, axis=0)
        sell_power = np.nansum(sell_power, axis=0)
        try:
            res = (buy_power - sell_power) / (buy_power + sell_power + 1e-4)
        except RuntimeWarning as e:
            return 0
        return res.item()


class HCVOLFactor(Factor):
    name = "HCVOL"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_buy_p = builder.get_recent_features(name="td_buy_price", window=window)
        td_buy_v = builder.get_recent_features(name="td_buy_vol", window=window)
        close = builder.get_recent_features(name="close", window=window)
        if len(close) < window:
            return 0

        v = np.where(td_buy_p > close.iloc[-1]+1e-5, td_buy_v, 0)
        try:
            res = np.nansum(v, axis=0) / np.nansum(td_buy_v, axis=0)
        except RuntimeWarning as e:
            return None
        return res.item()


class TradeAvgVolInHighPriceFactor(Factor):
    name = 'trade_avgvol_in_high_price'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        window_tot = window * 2 -1
        perc = self.kwargs["perc"]
        td_buy_v = builder.get_recent_features(name="td_buy_vol", window=window)
        td_sell_v = builder.get_recent_features(name="td_sell_vol", window=window)
        td_buy_n = builder.get_recent_features(name="td_buy_num", window=window)
        td_sell_n = builder.get_recent_features(name="td_sell_num", window=window)
        close = builder.get_recent_features(name="close", window=window_tot)
        if len(close) < window:
            return 0

        # size = len(close) - window + 1
        # vol = (td_buy_v + td_sell_v)[-size:].reset_index(drop=True)
        # n = (td_buy_n + td_sell_n)[-size:].reset_index(drop=True)
        # prank = close.rolling(window=window).rank(pct=True)[-window:].reset_index(drop=True)
        # v_in_region = vol[prank > perc].sum()
        # n_in_region = n[prank > perc].sum()
        # v_total = vol.sum()
        # n_total = n.sum()

        vol = (td_buy_v + td_sell_v)
        n = (td_buy_n + td_sell_n)
        prank = close.rolling(window=window).rank(pct=True)[-window:].reset_index(drop=True)
        v_in_region = vol[prank > perc + 1e-7].sum()
        n_in_region = n[prank > perc + 1e-7].sum()
        v_total = vol.sum()
        n_total = n.sum()

        try:
            res = (v_in_region / n_in_region) / (v_total / n_total)
        except RuntimeWarning as e:
            return 0
        return res.item()



class EnPowerFactor(Factor):
    name = 'bs_power_rough'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        buy_v = builder.get_recent_features(name="en_buy_vol", window=window)
        buy_p = builder.get_recent_features(name="en_buy_price", window=window)
        sell_v = builder.get_recent_features(name="en_sell_vol", window=window)
        sell_p = builder.get_recent_features(name="en_sell_price", window=window)
        close = builder.get_recent_features(name="close", window=1)

        if len(buy_v) < window:
            return None

        close = close.item()
        buy_power = buy_v * buy_p / close
        sell_power = sell_v * (2*close - sell_p) / close
        buy_power = np.nansum(buy_power, axis=0)
        sell_power = np.nansum(sell_power, axis=0)
        res = (buy_power - sell_power) / (buy_power + sell_power + 1e-4)
        return res.item()

class PriceVolCorrFactor(Factor):
    name = 'pv_corr'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        close = builder.get_recent_features(name="close", window=window)
        vol = builder.get_recent_features(name="td_vol", window=window)

        if len(close) < int(window/2):
            return 0

        res = fill_na(ts_corr(close, vol, window=window))
        return res.iloc[-1].item()


class RetVolCorrFactor(Factor):
    name = 'ret_avgvol_corr'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        close = builder.get_recent_features(name="close", window=window+1)
        vol = builder.get_recent_features(name="td_vol", window=window)
        td_buy_num = builder.get_recent_features(name="td_buy_num", window=window)
        td_sell_num = builder.get_recent_features(name="td_sell_num", window=window)

        if len(close) < int(window/2)+1:
            return 0
        size = len(vol)
        ret_v = ret(close)[-size:].reset_index(drop=True)
        res = ts_corr(ret_v, fill_na(vol/(td_buy_num + td_sell_num)), window=window)
        res.fillna(0, inplace=True)
        return res.iloc[-1].item()


class TradeSellFactor(Factor):
    name = 'td_sell_rank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_sell_p = builder.get_recent_features(name="td_sell_price", window=window)
        td_sell_v = builder.get_recent_features(name="td_sell_vol", window=window)

        if len(td_sell_p) < int(window/3):
            return None

        res = fill_na(td_sell_p * td_sell_v)
        res = ts_rank(res, window)
        return res.iloc[-1].item()


class RetSkewFactor(Factor):
    name = 'ret_skew'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        shift = self.kwargs["shift"]
        window_tot = window + shift
        close = builder.get_recent_features(name="close", window=window_tot)

        if len(close) < int(window/3)+shift:
            return None

        close = close.round(4)
        ret_v = ret(close, shift)
        res = ts_skew(ret_v, window)
        return res.iloc[-1].item()


class EnBuyPriceStdFactor(Factor):
    name = 'en_b_price_std'
    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window_tot = window1 + window2 - 1
        en_buy_p = builder.get_recent_features(name="en_buy_price", window=window_tot)
        if len(en_buy_p) < int((window1+window2)/3)-1:
            return None

        std_v = ts_std(en_buy_p, window2)
        res = ts_rank(std_v, window1)
        return res.iloc[-1].item()


class AwesomeOscillatorFactor(Factor):
    name = 'AwesomeOsc'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        window_tot = window + 34
        low = builder.get_recent_features(name="low", window=None)
        high = builder.get_recent_features(name="high", window=None)

        if len(low) < int(window/3) + 34 - 1:
            return None

        ts_high = ts_max(high, window)
        ts_low = ts_min(low, window)
        mean = (ts_high + ts_low) / 2
        res = ta.SMA(mean, 5) - ta.SMA(mean, 34)
        return res.iloc[-1].item()

class AlligatorTeethFactor(Factor):
    name = 'AlligatorTeeth'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        shift = self.kwargs["shift"]
        window_tot = window + shift + 8
        low = builder.get_recent_features(name="low", window=None)
        high = builder.get_recent_features(name="high", window=None)

        if len(low) < int(window/3) + shift + 8 - 1:
            return None

        ts_high = ts_max(high, window)
        ts_low = ts_min(low, window)
        mean = (ts_high + ts_low) / 2
        res = ta.SMA(mean, 8)
        res = ret(res, shift)
        return res.iloc[-1].item()

class RetKurtFactor(Factor):
    name = 'ret_kurt'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        window = self.kwargs["window"]
        window_tot = window + shift
        close = builder.get_recent_features(name="close", window=window_tot)
        if len(close) < int(window/3) + shift:
            return 0

        ret_v = ret(close, shift)
        res = ts_kurtosis(ret_v, window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else res.iloc[-1]


class LCVOLFactor(Factor):
    name = 'LCVOL'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_sell_p = builder.get_recent_features(name="td_sell_price", window=window)
        td_sell_v = builder.get_recent_features(name="td_sell_vol", window=window)
        close = builder.get_recent_features(name="close", window=window)
        if len(close) < window:
            return 0

        td_sell_p.ffill(inplace=True)
        new_v = np.where(td_sell_p < close.iloc[-1] - 1e-5, td_sell_v, 0)
        try:
            res = np.nansum(new_v, axis=0) / np.nansum(td_sell_v, axis=0)
        except RuntimeWarning as e:
            return 0
        return res.item()

class TradeEnVolRatioFactor(Factor):
    name = 'td_and_en_vol_ratio_sell_dir'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        td_sell_v = builder.get_recent_features(name="td_sell_vol", window=window)
        en_sell_v = builder.get_recent_features(name="en_sell_vol", window=window)

        if len(en_sell_v) < int(window/3):
            return 0

        td_v = ts_sum(td_sell_v, window)
        en_v = ts_sum(en_sell_v, window)
        res = fill_na(td_v / en_v)
        return res.iloc[-1].item()

class AroonDownFactor(Factor):
    name = 'AroonDown'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        timeperiod = self.kwargs["timeperiod"]
        window_tot = window + timeperiod

        low = builder.get_recent_features(name="low", window=None)
        high = builder.get_recent_features(name="high", window=None)
        if len(low) < int(window/3) + timeperiod:
            return None

        ts_low = ts_min(low, window)
        ts_high = ts_max(high, window)

        res = ta.AROON(high=ts_high, low=ts_low, timeperiod=timeperiod)[1]
        return res.iloc[-1].item()

class TradePriceStdFactor(Factor):
    name = 'td_price_std'

    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window_tot = window1 + window2
        vwap = builder.get_recent_features(name="vwap", window=window_tot)
        if len(vwap) < int(window_tot/3) - 1:
            return None

        res = ts_std(vwap.ffill(), window2)
        res = ts_rank(res, window1)
        return res.iloc[-1].item()


class TradeNumInLowPriceFactor(Factor):
    name = 'trade_num_in_low_price'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        window_tot = window + window - 1
        perc = self.kwargs["perc"]
        td_buy_n = builder.get_recent_features(name="td_buy_num", window=window)
        td_sell_n = builder.get_recent_features(name="td_sell_num", window=window)
        close = builder.get_recent_features(name="close", window=window_tot)
        if len(close) < window:
            return 0

        # size = len(close) - window + 1
        # num = (td_buy_n + td_sell_n)[-size:].reset_index(drop=True)
        # prank = close.rolling(window=window).rank(pct=True)[-size:].reset_index(drop=True)

        num = (td_buy_n + td_sell_n)
        prank = close.rolling(window=window).rank(pct=True)[-window:].reset_index(drop=True)

        new_num = num.where(prank < perc - 1e-7, 0)
        n_in_region = new_num.rolling(window=window, min_periods=1).sum()
        n_total = num.rolling(window=window, min_periods=1).sum()

        try:
            res = n_in_region / n_total
        except RuntimeError as e:
            return 0
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class CancelTimeBuyMedFactor(Factor):
    name = 'ct_b_med_tsrank'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        ct_b_med = builder.get_recent_features(name="cancel_buy_time_med", window=window)

        if len(ct_b_med) < int(window/3):
            return None

        inv_t = np.where(ct_b_med == 0, 0, 1/ct_b_med)
        res = ts_rank(pd.Series(inv_t), window=window)
        return res.iloc[-1].item()


class EnVolInbalanceFactor(Factor):
    name = 'en_v_order_inbalance'

    def compute(self, builder, **kwargs):
        en_buy_v = builder.get_recent_features(name="en_buy_vol", window=1)
        en_sell_v = builder.get_recent_features(name="en_sell_vol", window=1)
        if len(en_buy_v) < 1:
            return None
        res = fill_na((en_buy_v - en_sell_v) / (en_buy_v + en_sell_v))
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None




