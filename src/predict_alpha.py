import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

# setup
MODEL_DIR = Path('output/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = Path('output/plots')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# load
df = pd.read_parquet('data/processed/daily_trader_metrics.parquet')
if len(df) < 500:
    print("Low data volume.")

# feature engineering
df = df.sort_values(['account', 'date'])

# lag metrics
df['prev_sentiment'] = df.groupby('account')['value'].shift(1)
df['prev_volatility'] = df.groupby('account')['value'].transform(lambda x: x.rolling(2).std()).shift(1)
df['momentum_3d'] = df.groupby('account')['net_pnl'].transform(lambda x: x.rolling(3).mean()).shift(1)

# target finding
df['next_pnl'] = df.groupby('account')['net_pnl'].shift(-1)
df['target'] = (df['next_pnl'] > 0).astype(int)

# drop na
model_data = df.dropna()

features = [
    'value', 'net_pnl', 'trade_volume', 'win_rate', 
    'avg_position_size', 'leverage_p95', 'long_short_ratio',
    'prev_sentiment', 'prev_volatility', 'momentum_3d'
]
X = model_data[features]
y = model_data['target']

print(f"training on {len(X)} samples. Base rate: {y.mean():.1%}")

# pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1))
])

tscv = TimeSeriesSplit(n_splits=3)
scores = cross_val_score(pipe, X, y, cv=tscv, scoring='precision')

# fit full model
pipe.fit(X, y)

# results
print("\nResults")
print(f"Precision: {scores.mean():.1%} (Range: {scores.min():.1%} - {scores.max():.1%})")

# feature importance
imps = pd.DataFrame({
    'feature': features,
    'importance': pipe.named_steps['rf'].feature_importances_
}).sort_values('importance')

print(f"Top Predictor: {imps.iloc[-1]['feature']}")

# save plot
plt.figure(figsize=(10, 8))
plt.barh(imps['feature'], imps['importance'])
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'feature_importance.png')

# save model
joblib.dump(pipe, MODEL_DIR / 'sentiment_alpha_v1.pkl')
print("Model saved.")