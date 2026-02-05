import pandas as pd
import numpy as np
from pathlib import Path

# setup paths
RAW_DIR = Path('data/raw')
PROC_DIR = Path('data/processed')
PROC_DIR.mkdir(parents=True, exist_ok=True)

# load raw data
trades = pd.read_csv(RAW_DIR / 'historical_data.csv')
sentiment = pd.read_csv(RAW_DIR / 'fear_greed_index.csv')

print(f"trades shape: {trades.shape}")
print(f"sentiment shape: {sentiment.shape}")

# preprocess trades
trades['timestamp_utc'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M')
trades['date'] = trades['timestamp_utc'].dt.normalize()

# type casting to save memory
for col in ['Side', 'Direction', 'Crossed']:
    trades[col] = trades[col].astype('category')

# numeric cleanup
trades['Closed PnL'] = pd.to_numeric(trades['Closed PnL'], errors='coerce')
trades['Size USD'] = pd.to_numeric(trades['Size USD'], errors='coerce')

# preprocess sentiment
max_ts = sentiment['timestamp'].max()
unit = 'ms' if max_ts > 3e10 else 's'
sentiment['timestamp_utc'] = pd.to_datetime(sentiment['timestamp'], unit=unit)
sentiment['date'] = pd.to_datetime(sentiment['date'])
sentiment['classification'] = sentiment['classification'].astype('category')

# aggregation logic
print("aggregating daily metrics...")

# filter junk
valid_trades = trades[trades['Size USD'] >= 0].copy()

# remove zombie accounts (0 variance)
stats = valid_trades.groupby('Account').agg({'Closed PnL': 'var', 'Size USD': 'var'})
zombies = stats[(stats['Closed PnL'] == 0) | (stats['Size USD'] == 0)].index
valid_trades = valid_trades[~valid_trades['Account'].isin(zombies)]

# group by day + account
daily = valid_trades.groupby(['date', 'Account']).agg({
    'Closed PnL': ['sum', 'count', lambda x: (x > 0).mean()],
    'Size USD': ['mean', lambda x: np.percentile(x, 95) if len(x) > 0 else 0],
    'Side': lambda x: (x == 'BUY').sum() / (x == 'SELL').sum() if (x == 'SELL').sum() > 0 else np.inf
}).reset_index()

# clean up columns
daily.columns = ['date', 'account', 'net_pnl', 'trade_volume', 'win_rate', 
                 'avg_position_size', 'leverage_p95', 'long_short_ratio']
daily['long_short_ratio'] = daily['long_short_ratio'].replace([np.inf, -np.inf], np.nan)

print(f"rows after grouping: {len(daily)}")

# merge & save
# left join to keep all trade data, then forward fill sentiment
merged = daily.merge(sentiment[['date', 'value', 'classification']], on='date', how='left')
merged = merged.sort_values(['account', 'date'])
merged['value'] = merged.groupby('account')['value'].ffill()
merged['classification'] = merged.groupby('account')['classification'].ffill()

out_file = PROC_DIR / 'daily_trader_metrics.parquet'
merged.to_parquet(out_file, index=False)

print(f"DONE. Saved {len(merged):,} rows to {out_file}")