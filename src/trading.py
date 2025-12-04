# trading.py
# ---------------------------------------------------------
# PnL calculation using project-specified formula:
# BUY  → pnl -= close
# SELL → pnl += close
# ---------------------------------------------------------

import pandas as pd
import numpy as np


def compute_pnl(X_test_unscaled, predictions):
    results = X_test_unscaled[['Close']].copy()
    results['Predicted'] = predictions
    results['model_call'] = results['Predicted'].apply(lambda x: 'buy' if x == 1 else 'sell')

    pnl = 0
    pnl_list = []

    for i in range(len(results)):
        close_price = results.iloc[i]['Close']
        call = results.iloc[i]['model_call']

        if call == 'buy':
            pnl -= close_price
        else:
            pnl += close_price

        pnl_list.append(pnl)

    results['model_pnl'] = pnl_list
    return results
