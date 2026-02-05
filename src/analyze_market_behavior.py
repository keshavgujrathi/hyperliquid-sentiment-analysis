import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# config
PLOT_DIR = Path('output/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# load
df = pd.read_parquet('data/processed/daily_trader_metrics.parquet')
print(f"records: {len(df):,}")

# PnL Distribution
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='classification', y='net_pnl')
plt.ylim(-500, 500) # remove outliers
plt.title('PnL Distribution by Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'sentiment_pnl_distribution.png', dpi=300)
plt.close()

# Leverage Bar Chart
plt.figure(figsize=(10, 6))
lev_data = df.groupby('classification', observed=True)['leverage_p95'].mean().reset_index()
sns.barplot(data=lev_data, x='classification', y='leverage_p95')
plt.title('Avg 95th Percentile Leverage by Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'leverage_by_sentiment.png', dpi=300)
plt.close()

# Strategy

print("\nActionable Strategies")

# get simple averages
avgs = df.groupby('classification', observed=True)[['net_pnl', 'leverage_p95']].mean()
fear_pnl = avgs.loc['Fear', 'net_pnl'] if 'Fear' in avgs.index else 0
greed_pnl = avgs.loc['Greed', 'net_pnl'] if 'Greed' in avgs.index else 0
fear_lev = avgs.loc['Fear', 'leverage_p95'] if 'Fear' in avgs.index else 0

# Strategy 1: PnL Diff
if fear_pnl > greed_pnl:
    diff = ((fear_pnl - greed_pnl) / greed_pnl) * 100
    print(f"Strategy 1: The Fear Premium. Allocate more capital when Sentiment < 40.")
    print(f"   -> Fear outperforms Greed by {diff:.1f}% (${fear_pnl:.0f} vs ${greed_pnl:.0f})")
else:
    print(f"Strategy 1: Momentum Riding. Greed is outperforming Fear.")

# Strategy 2: Leverage Check
if fear_lev > 20000:
    print(f"Strategy 2: Volatility Dampening. Enforce leverage caps during Fear.")
    print(f"   -> Avg leverage hits {fear_lev:,.0f}x during panic.")

print("\nInsights")
print(f"Best Regime: {avgs['net_pnl'].idxmax()} (${avgs['net_pnl'].max():.2f})")
print(f"High Risk Regime: {avgs['leverage_p95'].idxmax()} ({avgs['leverage_p95'].max():.0f}x lev)")