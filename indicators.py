import numpy as np


def MovingAverage(N, close_prices_arr):

    ma_arr = [0 for _ in range(N-1)]

    for index in range(len(close_prices_arr)-N+1):
        ma_arr.append(sum(close_prices_arr[index:N+index])/N)

    return ma_arr


def ExponentialMovingAverage(N, close_prices_arr):

    ma_arr = [0 for _ in range(N-1)]

    for index in range(len(close_prices_arr)-N+1):
        ma_arr.append(
            (sum(close_prices_arr[index:N+index-1]) + 2*close_prices_arr[N+index-1]) / N)

    return ma_arr


def MACD(close_prices_arr):
    ema_12 = ExponentialMovingAverage(N=12, close_prices_arr=close_prices_arr)
    ema_26 = ExponentialMovingAverage(N=26, close_prices_arr=close_prices_arr)
    macd = np.array(ema_12) - np.array(ema_26)
    signal = ExponentialMovingAverage(N=9, close_prices_arr=macd)
    # histogram = macd - np.array(signal)
    return macd, signal
