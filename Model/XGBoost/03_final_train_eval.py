import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json, time
import matplotlib.pyplot as plt

df = pd.read_pickle('features.pkl')
df = df.sort_values('timestamp').reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in ('timestamp', 'demand_mw')]
n = len(df)
test_size = int(n * 0.10)
train_end = n - test_size

X_train, X_test = df[feature_cols].iloc[:train_end], df[feature_cols].iloc[train_end:]
y_train, y_test = df['demand_mw'].iloc[:train_end], df['demand_mw'].iloc[train_end:]

best_params = json.load(open('best_params.json'))
print("Using params:", best_params)

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    eval_metric='mae',
    **best_params
)

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print(f"Final fit done in {time.time()-t0:.1f}s")

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = float(np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100)

print(f"\n=== Test set performance ({df['timestamp'].iloc[train_end]} to {df['timestamp'].iloc[-1]}) ===")
print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")
print(f"Mean actual demand in test set: {y_test.mean():.2f} MW")

# Feature importance (top 20)
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 20 features by importance:")
print(importances.head(20))

# Save metrics
metrics = {
    'mae': mae, 'rmse': rmse, 'mape_pct': mape,
    'test_rows': len(X_test), 'train_rows': len(X_train),
    'test_start': str(df['timestamp'].iloc[train_end]),
    'test_end': str(df['timestamp'].iloc[-1]),
    'best_params': best_params,
}
json.dump(metrics, open('metrics.json', 'w'), indent=2)

# Export model binary (native XGBoost format - reloadable via xgb.XGBRegressor().load_model())
model.save_model('xgb_demand_model.json')
model.get_booster().save_model('xgb_demand_model.ubj')

# Save feature list (needed to reconstruct input order at inference time)
json.dump(feature_cols, open('feature_columns.json', 'w'), indent=2)
importances.to_csv('feature_importance.csv', header=['importance'])

print("\nSaved: xgb_demand_model.json, xgb_demand_model.ubj, feature_columns.json, metrics.json, feature_importance.csv")

# plot actual vs predicted
plt.figure(figsize=(15, 6))

plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted")

plt.xlabel("Time")
plt.ylabel("Demand (MW)")
plt.title("Actual vs Predicted Electricity Demand")
plt.legend()

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()