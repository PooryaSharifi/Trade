import datetime as dt
import pandas as pd
import numpy as np
import time
# import requests
import glob
import os.path
# from elmo import build_model, LMDataGenerator, DATA_SET_DIR, MODELS_DIR, parameters as elmo_parameters
# import re


def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean()

    # Create MACD
    dataset['26ema'] = dataset['Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

    # Create Exponential moving average
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean()

    # Create Momentum
    dataset['momentum'] = dataset['Close'] - 1
    return dataset


def get_fundamental_indicators(dataset):
    tweets = pd.read_csv('../refile.csv', header=None)
    tweets.sort_values([4, 5], inplace=True)
    tweets = tweets[[4, 35]]
    dates = list(tweets[4])
    feels = list(tweets[35])

    q = []
    dpi = []
    pnt = 0
    for date in df['Date']:
        while date >= dates[pnt]:
            q.append([dates[pnt], feels[pnt]])
            pnt += 1
        p = 0
        while len(q) - p > 3 and (dt.datetime.strptime(f'{date} 00:00:00', "%Y-%m-%d %H:%M:%S") - dt.datetime.strptime(
                f'{q[p][0]} 00:00:00', "%Y-%m-%d %H:%M:%S")).days > 2:
            p += 1
        q = q[p:]

        dpi.append(sum([d_f[1] / (len(q) + i) for i, d_f in enumerate(q)]))
    df['Dpi'] = np.array(dpi)


def trendiness(symbol, trend):
    dates = trend['Date'].tolist()
    trend = trend['Trend'].tolist()
    pnt = 0
    trends = []
    for date in symbol['Date']:
        while pnt < len(dates) and dates[pnt] <= date:
            pnt += 1
        trends.append(trend[pnt - 1])
    symbol['Trend'] = trends


def feeler(symbol, tweets):
    # TODO only the cols
    # replies_count,retweets_count,likes_count
    tweets['Count'] = 1
    tweets = tweets.groupby('date').sum()
    dates = tweets.index.tolist()
    likes = tweets['likes_count'].tolist()
    feels = tweets['Feel'].tolist()
    counts = tweets['Count'].tolist()
    feelings, trends = [], []
    pnt = 0
    for date in symbol['Date']:
        while pnt < len(dates) and dates[pnt] <= date:
            pnt += 1
        trends.append(likes[pnt - 1] ** .5 + counts[pnt - 1])
        feelings.append(feels[pnt - 1] / counts[pnt - 1])
    symbol['Like'] = trends
    symbol['Feel'] = feelings


window_size = 66
k = 2
x_dim = 6
skip = 6


def normalise(window, symbol, date, limit):
    _min = window[: limit].min()
    scale = window[: limit].max() - _min
    _n = (window - _min) / scale if scale else window - _min
    return np.concatenate([(symbol, date, _min, scale), _n], axis=0).astype(np.float32)

# def normalise(window, symbol, date, limit):
#     _min = window.min()
#     scale = window.max() - _min
#     _n = (window - _min) / scale if scale else window - _min
#     return np.concatenate([(symbol, date, _min, scale), _n], axis=0)


def normalize(window, symbol, date, limit=window_size - k):
    return np.apply_along_axis(normalise, 0, window, symbol=symbol, date=date, limit=limit).astype(np.float32)


def denormalise(window):
    _min = window[0]
    scale = window[1]
    return (window[2:] * scale if scale else window[2:]) + _min


def denormalize(window):
    return np.apply_along_axis(denormalise, 0, window).astype(np.float32)


# def x(df, skip=6):
#     df = df.interpolate(method='linear', axis=0).ffill().bfill()
#     df = get_technical_indicators(df)
#     df = df[cols].values[skip:]
#     return np.stack(normalize(df[i: i + window_size, :]) for i in range(df.shape[0] - window_size + 1))

# def time_window_data(df, cols, symbol_name, skip=6):
#     df = df.interpolate(method='linear', axis=0).ffill().bfill()
#     df = get_technical_indicators(df)
#     shift = df['Close'].shift(1)
#     df['Low_Ret'] = (shift - df['Low']) / shift
#     df['High_Ret'] = (df['High'] - shift) / shift
#     df['Close_Ret'] = (df['Close'] - shift) / shift
#     dates = df['Date']
#     df = df[cols].values
#     return np.stack(normalize(df[i: i + window_size, :], symbol=symbol_name, date=int(dates[i + window_size - k - 1].replace('-', ''))) for i in range(df.shape[0] - window_size + 1))[skip:]
#
# X = sorted([time_window_data(symbol, cols, symbol_name=i).astype(np.float32) for i, symbol in enumerate(symbols)], key=lambda x: -x.shape[0])
# _X = []
# while X:
#     seg = X[-1].shape[0]
#     _X.append(np.zeros((seg * len(X), window_size + 4, len(cols))))
#     for i, x in enumerate(X):
#         _X[-1][i::len(X)] = x[-seg:]
#         X[i] = x[:-seg]
#     while X and not X[-1].shape[0]:
#         X.pop()
# X = np.concatenate(list(reversed(_X)), axis=0)
# print(X.shape)

def time_window_data(df, cols, symbol_name, skip=6, interpolate=False):
    dates = df['Date']
    df = df.interpolate(method='linear', axis=0).ffill().bfill()
    df = df[cols].values
    limit = window_size if interpolate else window_size - k
    return np.stack(normalize(df[i: i + window_size, :], symbol=symbol_name, date=int(dates[i + window_size - k].replace('-', '')[2:]), limit=limit) for i in range(df.shape[0] - window_size + 1)).astype(np.float32)[skip:]


def xxx(cols):
    if os.path.exists('X.npy') and os.path.exists('XX.npy'):
        return
    # mighty_ip = '192.168.1.52'
    # names = set(requests.get(f'http://{mighty_ip}:5000/symbols').json())
    names = glob.glob('Symbols/*.csv')
    print(names)
    # dominant_date = [name[-14:-4] for name in names if '-' in name[-14:-4]]
    # dominant_date = max(set(dominant_date), key=dominant_date.count)
    # dominant_date = '_' + dominant_date
    dominant_date = ''
    print(dominant_date)
    lengths = []
    global_dates = set()
    X = []
    symbol_names = list(sorted({name[8:13] for name in names}))
    symbol_its = {symbol: i for i, symbol in enumerate(symbol_names)}
    for symbol in symbol_names:
        if f'Symbols/{symbol}{dominant_date}.csv' in names \
                and f'Symbols/{symbol}_Trend{dominant_date}.csv' in names \
                and f'Symbols/{symbol}_Twitter{dominant_date}.csv' in names:
            t0 = time.time()
            # if not os.path.exists(f'Symbols/{symbol}_{dominant_date}.csv'):
            #     os.system(f'wget http://{mighty_ip}:5000/symbols/{symbol}_{dominant_date}.csv -O Symbols/{symbol}_{dominant_date}.csv')
            # if not os.path.exists(f'Symbols/{symbol}_Trend_{dominant_date}.csv'):
            #     os.system(f'wget http://{mighty_ip}:5000/symbols/{symbol}_Trend_{dominant_date}.csv -O Symbols/{symbol}_Trend_{dominant_date}.csv')
            # if not os.path.exists(f'Symbols/{symbol}_Twitter_{dominant_date}.csv'):
            #     os.system(f'wget http://{mighty_ip}:5000/symbols/{symbol}_Twitter_{dominant_date}.csv -O Symbols/{symbol}_Twitter_{dominant_date}.csv')
            df = pd.read_csv(f'Symbols/{symbol}{dominant_date}.csv')
            df = df[(df['Date'] >= '2016-01-01')]
            if len(df) >= window_size + skip and symbol != 'DRKH1':
                df = df.interpolate(method='linear', axis=0).ffill().bfill()
                global_dates = global_dates.union(set(df['Date']))
                lengths.append((symbol, len(df)))
                print(symbol)
                df = get_technical_indicators(df)
                shift = df['Close'].shift(1)
                df['Low_Ret'] = (shift - df['Low']) / shift
                df['High_Ret'] = (df['High'] - shift) / shift
                df['Close_Ret'] = (df['Close'] - shift) / shift
                trendiness(df, pd.read_csv(f'Symbols/{symbol}_Trend{dominant_date}.csv'))
                feeler(df, pd.read_csv(f'Symbols/{symbol}_Twitter{dominant_date}.csv'))
                dates = [np.float32(date.replace('-', '')[2:]) for date in df['Date']]
                df = df[cols].values
                x = np.stack(normalize(df[i: i + window_size, :], symbol=symbol_its[symbol], limit=window_size, date=dates[i + window_size - k - 1]) for i in range(df.shape[0] - window_size + 1))[skip:]
                X.append(x)
                print(time.time() - t0)
    X = np.concatenate(X)
    np.save('X.npy', X)

    print(X.shape)
    lengths = sorted(lengths, key=lambda x: -x[1])
    print(lengths[319])
    symbol_names = list(sorted([symbol for symbol, _ in lengths][:320]))
    print(symbol_names)
    print(len(symbol_names))
    XX = []
    for symbol in symbol_names:
        t0 = time.time()
        df = pd.read_csv(f'Symbols/{symbol}{dominant_date}.csv')
        df = df[(df['Date'] >= '2016-01-01')]
        if len(df) >= window_size + skip and symbol != 'DRKH1':
            for date in global_dates - set(df['Date']):
                df = df.append({'Date': date}, ignore_index=True)
                df = df.sort_values('Date')
            df = df.interpolate(method='linear', axis=0).ffill().bfill()
            print(symbol)
            df = get_technical_indicators(df)
            shift = df['Close'].shift(1)
            df['Low_Ret'] = (shift - df['Low']) / shift
            df['High_Ret'] = (df['High'] - shift) / shift
            df['Close_Ret'] = (df['Close'] - shift) / shift
            trendiness(df, pd.read_csv(f'Symbols/{symbol}_Trend{dominant_date}.csv'))
            feeler(df, pd.read_csv(f'Symbols/{symbol}_Twitter{dominant_date}.csv'))
            dates = [np.float32(date.replace('-', '')[2:]) for date in df['Date']]
            df = df[cols].values
            x = np.stack(normalize(df[i: i + window_size - k, :], symbol=symbol_its[symbol], limit=window_size - k, date=dates[i + window_size - k - 1]) for i in range(df.shape[0] - window_size + k + 1))[skip:]
            XX.append(x)
            print(time.time() - t0)
    XX = np.stack(XX, axis=2)
    print(np.max(XX[:, 4:, :, :]))
    print(XX.shape)
    np.save('XX.npy', XX)
