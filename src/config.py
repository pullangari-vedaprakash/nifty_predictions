# config.py
# ---------------------------------------------------------
# Configuration values used across the pipeline
# ---------------------------------------------------------

DATA_PATH = "data/nifty50_ticks.csv"

FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close',
    'SMA_5', 'SMA_20',
    'EMA_10', 'EMA_20',
    'RSI',
    'C_O', 'H_L',
    'upper_wick', 'lower_wick',
    'ret_1', 'ret_3', 'ret_5',
    'vol_10', 'vol_20',
    'bb_upper', 'bb_lower', 'bb_width',
    'macd', 'macd_signal', 'macd_hist',
    'ATR_14'
]

TRAIN_SPLIT_RATIO = 0.7
