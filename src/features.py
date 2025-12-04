# features.py
# ---------------------------------------------------------
# Feature engineering + target generation
# ---------------------------------------------------------

import pandas as pd
import numpy as np


def add_features(df):

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['C_O'] = df['Close'] - df['Open']
    df['H_L'] = df['High'] - df['Low']
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']

    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_3'] = df['Close'].pct_change(3)
    df['ret_5'] = df['Close'].pct_change(5)

    df['vol_10'] = df['ret_1'].rolling(10).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()

    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma20

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    df.dropna(inplace=True)
    return df
