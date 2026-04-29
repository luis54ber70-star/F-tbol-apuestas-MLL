import pandas as pd, requests, joblib, os, numpy as np
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

API_KEY = os.getenv('API_KEY')
LEAGUE_ID = 262 # Liga MX en API-Football
SEASON = 2024

def get_data():
    url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={SEASON}"
    r = requests.get(url, headers={"x-apisports-key": API_KEY})
    r.raise_for_status()
    fixtures = r.json()['response']

    rows = []
    for f in fixtures:
        if f['fixture']['status']['short']!= 'FT': continue
        h = f['teams']['home']['name']
        a = f['teams']['away']['name']
        gh, ga = f['goals']['home'], f['goals']['away']
        if gh is None or ga is None: continue
        if gh > ga: result = 1 # Local
        elif gh == ga: result = 0 # Empate
        else: result = 2 # Visitante
        rows.append({'date': f['fixture']['date'], 'home': h, 'away': a,
                     'gh': gh, 'ga': ga, 'result': result})

    df = pd.DataFrame(rows).sort_values('date')
    df['date'] = pd.to_datetime(df['date'])
    return df

def make_features(df):
    df = df.copy()
    for team in pd.unique(df[['home', 'away']].values.ravel('K')):
        mask_h = df['home'] == team
        mask_a = df['away'] == team
        df.loc[mask_h, 'home_gf_avg'] = df.loc[mask_h, 'gh'].shift().rolling(5, min_periods=1).mean()
        df.loc[mask_a, 'away_gf_avg'] = df.loc[mask_a, 'ga'].shift().rolling(5, min_periods=1).mean()
        df.loc[mask_h, 'home_ga_avg'] = df.loc[mask_h, 'ga'].shift().rolling(5, min_periods=1).mean()
        df.loc[mask_a, 'away_ga_avg'] = df.loc[mask_a, 'gh'].shift().rolling(5, min_periods=1).mean()

    df['goal_diff_avg'] = df['home_gf_avg'] - df['home_ga_avg'] - (df['away_gf_avg'] - df['away_ga_avg'])
    df = df.dropna()
    return df

def backtest_roi(model, X, y):
    probs = model.predict_proba(X)
    picks = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    mask = conf > 0.52
    if mask.sum() == 0: return 0
    wins = (picks[mask] == y[mask]).sum()
    bets = mask.sum()
    roi = (wins * 0.95 - (bets - wins)) / bets
    return roi

df = get_data()
if len(df) < 50:
    print("No hay suficientes datos históricos. Abortando.")
    exit()

df = make_features(df)
features = ['goal_diff_avg', 'home_gf_avg', 'home_ga_avg', 'away_gf_avg', 'away_ga_avg']
X, y = df[features], df['result']

tss = TimeSeriesSplit(n_splits=3)
scores, rois = [], []
for train_idx, val_idx in tss.split(X):
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict_proba(X.iloc[val_idx])
    scores.append(log_loss(y.iloc[val_idx], pred))
    rois.append(backtest_roi(model, X.iloc[val_idx], y.iloc[val_idx]))

print(f"LogLoss promedio: {np.mean(scores):.3f}")
print(f"ROI promedio: {np.mean(rois)*100:.2f}%")

os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Guarda si ROI > 2% o si no existe modelo previo
if np.mean(rois) > 0.02 or not os.path.exists('models/xg_model.pkl'):
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    model.fit(X, y)
    joblib.dump(model, 'models/xg_model.pkl')
    df.to_csv('data/historico.csv', index=False)
    print("Modelo guardado.")
else:
    print("Modelo descartado. ROI < 2%")
