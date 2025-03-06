import pandas as pd
import numpy as np


def move(x, shift: int = 1):
    return x.shift(shift)


def delta(x, shift):
    # try:
    #     res = x.diff(shift)
    # except TypeError as e:
    #     return 0
    return x.diff(shift)


def ret(x, shift: int = 1):
    return x / x.shift(shift)-1


# def rowrank(x):
#     return x.rank(axis=1, pct=True)


def ts_rank(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).rank(pct=True)


def ts_sum(x, window):
    return x.rolling(window, min_periods=int(window/3)).sum(engine='numba')


def ts_mean(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).mean(engine='numba')


def ts_std(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).std(engine='numba')


def ts_skew(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).skew()


def ts_kurtosis(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).kurt()


def ts_min(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).min()


def ts_max(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).max()


def ts_med(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).median()


def ts_count(x, window):
    return x.rolling(window=window, min_periods=int(window / 3)).count()


def log(x):
    return np.log(x)


def divide(x, y):
    return x.divide(y)


def inv(x):
    return 1 / x


def mul(x, y):
    return x * y


def sub(x, y):
    return x - y


def add(x, y):
    return x + y


def ts_corr(x, y, window):
    corrs = x.rolling(window=window, min_periods=int(window / 2)).corr(y)
    corrs.replace([np.inf, -np.inf], np.nan, inplace=True)
    return corrs


def avoid_zero(x, small_num: float = 1e-4):
    return x.where(x != 0.0, small_num)


def fill_na(x: pd.Series, fill_num: float = 0.):
    return x.fillna(fill_num)


def fill_na_v2(x: pd.Series, y):
    return x.fillna(y)


def avg_price(price, vol, window: int = 10):
    return ts_sum(price * vol, window) / ts_sum(vol, window)
