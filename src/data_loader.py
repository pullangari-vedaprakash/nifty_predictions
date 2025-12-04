# data_loader.py
# ---------------------------------------------------------
# Loads raw data and prepares base dataframe
# ---------------------------------------------------------

import pandas as pd
import numpy as np


def load_data(file_path):
    df = pd.read_csv(
        file_path,
        sep=",",
        skiprows=1,
        header=None,
    )

    df = df.iloc[:, [2, 3, 4, 5, 6]].copy()
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close']

    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df = df.set_index('Timestamp')
    df = df.sort_index()
    df.dropna(inplace=True)

    return df
