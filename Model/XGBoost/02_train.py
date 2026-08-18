import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time, json

df = pd.read_pickle('features.pkl')
df = df.sort_values('timestamp').reset_index(drop=True)

feature_cols = [c for c in df.columns if c not in ('timestamp', 'demand_mw')]

n = len(df)
test_size = int(n * 0.10)
train_end = n - test_size

X_train_full, X_test = df[feature_cols].iloc[:train_end], df[feature_cols].iloc[train_end:]
y_train_full, y_test = df['demand_mw'].iloc[:train_end], df['demand_mw'].iloc[train_end:]

print(f"Full train: {X_train_full.shape}, Test: {X_test.shape}")

# For hyperparameter SEARCH ONLY, use the most recent 3 years of the training set
# (recent patterns matter more, and this keeps the search fast)
search_window = 24*365*3
X_search = X_train_full.iloc[-search_window:]
y_search = y_train_full.iloc[-search_window:]
print(f"Search subset: {X_search.shape}")

param_dist = {
    'n_estimators': [200, 400],
    'max_depth': [4, 6],
    'learning_rate': [0.03, 0.08],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_weight': [1, 5],
    'reg_lambda': [1, 2],
}

base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
)

tscv = TimeSeriesSplit(n_splits=2)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=8,
    scoring='neg_mean_absolute_error',
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=1,
)

t0 = time.time()
search.fit(X_search, y_search)
print(f"Search done in {time.time()-t0:.1f}s")
print("Best params:", search.best_params_)
print("Best CV MAE:", -search.best_score_)

json.dump(search.best_params_, open('best_params.json', 'w'), indent=2)
print("Saved best_params.json")
