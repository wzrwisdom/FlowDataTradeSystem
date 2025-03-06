import numpy as np
from FlowDataTradeSystem.factor.factors.base import Factor
from FlowDataTradeSystem.factor.factor_func import *
import pandas as pd
from llvmlite.ir import builder
from sortedcontainers import SortedList
import talib as ta
import scipy
import warnings
warnings.simplefilter("error", RuntimeWarning)

class EnBuyP10Factor(Factor):
    name = 'en_b_p10_tsrank_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        b1 = builder.get_recent_features(name="b1", window=window)

        if len(b1) < int(window/3):
            return None

        res = ts_rank(b1, window=window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class EnSellSumPriceRatioFactor(Factor):
    name = 'en_s_sumprice_ratio_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        window_tot = window
        en_sell_p = builder.get_recent_features(name="s1", window=window_tot)
        en_sell_v = builder.get_recent_features(name="sv1_sum", window=window_tot)
        avg_p = builder.get_recent_features(name="bs_avg_price", window=1)

        if len(en_sell_p) < int((window_tot)/3):
            return None

        sell_avg_p = avg_price(en_sell_p, en_sell_v, window_tot)
        sell_avg_p.reset_index(inplace=True, drop=True)
        res = sell_avg_p / avg_p.item()
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class MACDFactor(Factor):
    name = 'MACD_future'

    def compute(self, builder, **kwargs):

        last = builder.get_recent_features(name="last", window=None)
        res = ta.MACD(last)[0]
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class WILLRFactor(Factor):
    name = 'WILLR_future'
    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        last = builder.get_recent_features(name="last", window=None)

        if len(last) < int(window/3):
            return None

        ts_low = ts_min(last, window)
        ts_high = ts_max(last, window)
        res = ta.WILLR(ts_high, ts_low, last)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class RSIFactor(Factor):
    name = 'RSI_future'

    def compute(self, builder, **kwargs):
        last = builder.get_recent_features(name="last", window=None)
        res = ta.RSI(last)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class DeridBid1Factor(Factor):
    name = 'DeriPBid1_future'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        b1 = builder.get_recent_features(name="b1", window=shift + 1)
        if len(b1) < shift+1:
            return None

        res = delta(b1, shift)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class APOFactor(Factor):
    name = 'APO_future'

    def compute(self, builder, **kwargs):
        matype = self.kwargs["matype"]
        last = builder.get_recent_features(name="last", window=None)
        res = ta.APO(last, fastperiod=5, slowperiod=13, matype=matype)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class WCLFactor(Factor):
    name = 'WCL_future'
    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        shift = self.kwargs["shift"]
        last = builder.get_recent_features(name="last", window=None)

        if len(last) < int(window/3):
            return None

        ts_low = ts_min(last, window)
        ts_high = ts_max(last, window)
        res = ta.WCLPRICE(ts_high, ts_low, last)
        res = ret(res, shift)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class MomentumFactor(Factor):
    name = 'Momentum_future'

    def compute(self, builder, **kwargs):
        last = builder.get_recent_features(name="last", window=None)

        res = ta.MOM(last)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class TRIMAFactor(Factor):
    name = 'TRIMA_future'

    def compute(self, builder, **kwargs):
        # timeperiod = self.kwargs["timeperiod"]
        shift = self.kwargs["shift"]
        last = builder.get_recent_features(name="last", window=None)

        res = ta.TRIMA(last, timeperiod=10)
        res = ret(res, shift)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class CLHDELTAFACTOR(Factor):
    name = 'clh_delta_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        shift = self.kwargs["shift"]
        window_tot = window + shift
        last = builder.get_recent_features(name="last", window=window_tot)
        if len(last) < window_tot:
            return None

        o = last.rolling(window=window, min_periods=min(100, int(window/2))).apply(lambda x: x[0], raw=True)
        h = last.rolling(window=window, min_periods=min(100, int(window/2))).max()
        l = last.rolling(window=window, min_periods=min(100, int(window/2))).min()
        c = last.rolling(window=window, min_periods=min(100, int(window/2))).apply(lambda x: x[-1], raw=True)

        res = ((h-c) - (c-l)) / (h-l)
        res.where((h-l)!=0, 0, inplace=True)
        res = delta(res, shift)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class CLHFACTOR(Factor):
    name = 'clh_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        window_tot = window
        last = builder.get_recent_features(name="last", window=window_tot)
        if len(last) < min(100, int(window/2)):
            return None

        o = last.rolling(window=window, min_periods=min(100, int(window/2))).apply(lambda x: x[0], raw=True)
        h = last.rolling(window=window, min_periods=min(100, int(window/2))).max()
        l = last.rolling(window=window, min_periods=min(100, int(window/2))).min()
        c = last.rolling(window=window, min_periods=min(100, int(window/2))).apply(lambda x: x[-1], raw=True)

        res = ((h-c) - (c-l)) / (h-l)
        res.where((h-l)!=0, 0, inplace=True)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class AlligatorLipsFactor(Factor):
    name = 'AlligatorLips_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        shift = self.kwargs["shift"]
        last = builder.get_recent_features(name="last", window=None)

        ts_high = ts_max(last, window)
        ts_low = ts_min(last, window)
        mean = (ts_high + ts_low) / 2
        res = ta.SMA(mean ,5)

        res = ret(res, shift)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class RetVolCorrFactor(Factor):
    name = "ret_v_corr_future"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        last = builder.get_recent_features(name="last", window=window+1)
        vol = builder.get_recent_features(name="vol", window=window)

        if len(last) < int(window/2)+1:
            return None
        size = len(vol)
        ret_v = ret(last)[-size:].reset_index(drop=True)
        res = ts_corr(ret_v, vol, window=window)
        if np.isnan(res.iloc[-1]):
            return None
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class OBPriceDeriFactor(Factor):
    name = 'OB_price_2derivative_future'

    def compute(self, builder, **kwargs):
        shift = self.kwargs["shift"]
        window_tot = 2*shift+1
        avg_p = builder.get_recent_features(name="bs_avg_price", window=window_tot)
        if len(avg_p) < window_tot:
            return None

        avg_p = avg_p.round(4)
        avg_p_shift = move(avg_p, shift)
        avg_p_twoshift = move(avg_p, shift * 2)

        res = (avg_p_twoshift + avg_p - 2 * avg_p_shift) / avg_p
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class BestVImbalanceFactor(Factor):
    name = 'best_v_imbalance_tsrank_future'

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
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class AroonOscFactor(Factor):
    name = 'AroonOsc_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        timeperiod = self.kwargs["timeperiod"]
        window_tot = window + timeperiod

        last = builder.get_recent_features(name="last", window=None)
        if len(last) < int(window/3) + timeperiod:
            return None

        ts_low = ts_min(last, window)
        ts_high = ts_max(last, window)

        res = ta.AROONOSC(high=ts_high, low=ts_low, timeperiod=timeperiod)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class STOCHRSIfastdFactor(Factor):
    name = 'STOCHRSIfastd_future'
    def compute(self, builder, **kwargs):
        last = builder.get_recent_features(name="last", window=None)
        res = ta.STOCHRSI(last)[1]
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class DetrendFactor(Factor):
    name = 'Detrend2_future'

    def compute(self, builder, **kwargs):
        last = builder.get_recent_features(name="last", window=None)
        try:
            res = scipy.signal.detrend(last)[-1]
            return res.item()
        except ValueError as e:
            return None


class STOCHRSIfastkFactor(Factor):
    name = 'STOCHRSIfastk_future'
    def compute(self, builder, **kwargs):
        last = builder.get_recent_features(name="last", window=None)
        res = ta.STOCHRSI(last)[0]
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class TradePVRatioFactor(Factor):
    name = 'td_p_v_ratio_future'

    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window = window1 + window2
        vwap = builder.get_recent_features(name="vwap", window=window)
        vol = builder.get_recent_features(name="vol", window=window-1)
        if len(vwap) < int(window1/3):
            return None
        delta_vwap = delta(vwap, window2)[-window1:].reset_index(drop=True)
        delta_vwap = delta_vwap.round(10)
        vol_sum = ts_sum(vol, window2)[-window1:].reset_index(drop=True)
        res = (delta_vwap / vol_sum)
        res.fillna(0, inplace=True)

        res = ts_rank(res, window1)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class TradeVolInHihgPriceFactor(Factor):
    name = 'trade_vol_in_high_price_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        perc = self.kwargs["perc"]
        window_tot = 2 * window - 1
        vol = builder.get_recent_features(name="vol", window=window)
        price = builder.get_recent_features(name="last", window=window_tot)
        if len(price) < window:
            return 0

        # size = len(price) - window + 1
        # vol = (td_buy_v + td_sell_v)[-size:].reset_index(drop=True)
        prank = price.rolling(window=window).rank(pct=True)[-window:].reset_index(drop=True)
        n_in_region = vol[prank > perc + 1e-7].sum()
        n_total = vol.sum()
        try:
            res = n_in_region / n_total
        except RuntimeWarning as e:
            return 0
        return res.item()

class TradeRetVProdFactor(Factor):
    name = "td_ret_v_prod_future"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        vwap = builder.get_recent_features(name="vwap", window=window + 1)
        vol = builder.get_recent_features(name="vol", window=window)
        if len(vol) < int(window/3):
            return None
        res = ret(vwap) * vol
        res = ts_rank(res, window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class PriceVolCorrFactor(Factor):
    name = 'pv_corr_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        last = builder.get_recent_features(name="last", window=window)
        vol = builder.get_recent_features(name="vol", window=window)

        if len(last) < int(window/2):
            return 0

        res = fill_na(ts_corr(last, vol, window=window))
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class KeltnerChannelLowerFactor(Factor):
    name = 'KeltnerChannelLower_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        timeperiod1 = self.kwargs["timeperiod1"]
        timeperiod2 = self.kwargs["timeperiod2"]
        last = builder.get_recent_features(name="last", window=None)

        if len(last) < timeperiod1:
            return None

        ts_high = ts_max(last, window)
        ts_low = ts_min(last, window)
        atr = ta.ATR(high=ts_high, low=ts_low, close=last, timeperiod=timeperiod1)
        ema = ta.EMA(last, timeperiod2)
        res = ((ema - 2*atr) - last) / ema

        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class BSPowerFactor(Factor):
    name = 'bs_power_rough_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        buy_v = builder.get_recent_features(name="bv1_sum", window=window)
        buy_p = builder.get_recent_features(name="b1", window=window)
        sell_v = builder.get_recent_features(name="sv1_sum", window=window)
        sell_p = builder.get_recent_features(name="s1", window=window)
        last = builder.get_recent_features(name="last", window=1)

        if len(buy_v) < window:
            return None

        close = last.item()
        buy_power = buy_v * buy_p / close
        sell_power = sell_v * (2*close - sell_p) / close
        buy_power = np.nansum(buy_power, axis=0)
        sell_power = np.nansum(sell_power, axis=0)
        res = (buy_power - sell_power) / (buy_power + sell_power + 1e-4)
        return res.item() if not np.isnan(res) else None


class EnBSPriceDiffFactor(Factor):
    name = 'en_bs_price_diff_future'

    def compute(self, builder, **kwargs):
        b1 = builder.get_recent_features(name="b1", window=1)
        s1 = builder.get_recent_features(name="s1", window=1)
        avg_p = builder.get_recent_features(name="bs_avg_price", window=1)

        res = (b1 - s1) / avg_p
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class DPOFactor(Factor):
    name = "DPO_future"

    def compute(self, builder, **kwargs):
        timeperiod = self.kwargs["timeperiod"]
        last = builder.get_recent_features(name="last", window=None)
        if len(last) < int(timeperiod/2):
            return None

        res = move(last, int(timeperiod/2) + 1) - ta.SMA(last, timeperiod)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class LIFactor(Factor):
    name = "LI_future"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        last = builder.get_recent_features(name="last", window=window)

        if len(last) < int(window/3):
            return None
        res = ts_std(last, window) / ts_mean(last, window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class EnBuyPriceStdFactor(Factor):
    name = 'en_b_price_std_future'
    def compute(self, builder, **kwargs):
        window1 = self.kwargs["window1"]
        window2 = self.kwargs["window2"]
        window_tot = window1 + window2 - 1
        en_buy_p = builder.get_recent_features(name="b1", window=window_tot)
        if len(en_buy_p) < int((window1+window2)/3)-1:
            return None

        std_v = ts_std(en_buy_p, window2)
        res = ts_rank(std_v, window1)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class EnSellPriceFactor(Factor):
    name = "en_s_price_tsrank_future"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        s1 = builder.get_recent_features(name="s1", window=window)
        last = builder.get_recent_features(name="last", window=window)

        if len(last) < int(window/3):
            return None
        last = last.round(5)
        s1 = s1.round(4)
        res = ts_rank(s1 - last, window=window)
        return res.iloc[-1].item()

class ATRFactor(Factor):
    name = 'ATR_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        timeperiod = self.kwargs["timeperiod"]
        last = builder.get_recent_features(name="last", window=None)

        if len(last) < timeperiod:
            return None

        ts_high = ts_max(last, window)
        ts_low = ts_min(last, window)
        res = ta.ATR(high=ts_high, low=ts_low, close=last, timeperiod=timeperiod)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class BSPV1TsrankFactor(Factor):
    name = 'bs_pv1_tsrank_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        b1 = builder.get_recent_features(name="b1", window=window)
        s1 = builder.get_recent_features(name="s1", window=window)
        bv1 = builder.get_recent_features(name="bv1_sum", window=window)
        sv1 = builder.get_recent_features(name="sv1_sum", window=window)

        if len(b1) < int(window/3):
            return None
        res = ts_rank((b1 * bv1 - s1*sv1), window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None


class VLIFactor(Factor):
    name = "VLI_future"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        vol = builder.get_recent_features(name="vol", window=window)

        if len(vol) < int(window / 3):
            return None
        res = ts_std(vol, window) / ts_mean(vol, window)
        res.fillna(0, inplace=True)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class EnBuyPriceFactor(Factor):
    name = 'en_b_price_tsrank_future'

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        b1 = builder.get_recent_features(name="b1", window=window)
        last = builder.get_recent_features(name="last", window=window)

        if len(last) < int(window/3):
            return None
        res = ts_rank(b1 - last, window)
        return res.iloc[-1].item() if not np.isnan(res.iloc[-1]) else None

class ReturnVolProductFactor(Factor):
    name = "ret_v_prod_future"

    def compute(self, builder, **kwargs):
        window = self.kwargs["window"]
        close = builder.get_recent_features(name="last", window=window + 1)
        vol = builder.get_recent_features(name="vol", window=window)
        if len(close) < int(window / 3) + 1:
            return None

        previous_list = self.state.get("previous_list", None)
        if previous_list is None:
            previous_list = []
            # 计算收益率
            values = [
                (close.iloc[i + 1] - close.iloc[i]) / close.iloc[i] * vol.iloc[i + 1]
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


