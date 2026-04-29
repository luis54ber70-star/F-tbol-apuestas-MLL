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
    fixtures = r.json()['response']

    rows = []
    for f in fixtures:
        if f['fixture']['status']['short']!= 'FT': continue
        h = f['teams']['home']['name']
        a = f['teams']['away']['name']
        gh, ga = f['goals']['home'], f['goals']['away']
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
        df.loc[mask_h, 'home_gf_avg'] = df.loc[mask_h, 'gh'].shift().rolling(5).mean()
        df.loc[mask_a, 'away_gf_avg'] = df.loc[mask_a, 'ga'].shift().rolling(5).mean()
        df.loc[mask_h, 'home_ga_avg'] = df.loc[mask_h, 'ga'].shift().rolling(5).mean()
        df.loc[mask_a, 'away_ga_avg'] = df.loc[mask_a, 'gh'].shift().rolling(5).mean()

    df['goal_diff_avg'] = df['home_gf_avg'] - df['home_ga_avg'] - (df['away_gf_avg'] - df['away_ga_avg'])
    df = df.dropna()
    return df

def backtest_roi(model, X, y):
    probs = model.predict_proba(X)
    # Estrategia simple: apostar al que dé >52% prob
    picks = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    mask = conf > 0.52
    wins = (picks == y)[mask].sum()
    bets = mask.sum()
    if bets == 0: return 0
    # Asumiendo cuota promedio 1.95 para >52%
    roi = (wins * 0.95 - (bets - wins)) / bets
    return roi

df = get_data()
df = make_features(df)
features = ['goal_diff_avg', 'home_gf_avg', 'home_ga_avg', 'away_gf_avg', 'away_ga_avg']
X, y = df[features], df['result']

tss = TimeSeriesSplit(n_splits=3)
scores, rois = [], []
for train_idx, val_idx in tss.split(X):
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict_proba(X.iloc[val_idx])
    scores.append(log_loss(y.iloc[val_idx], pred))
    rois.append(backtest_roi(model, X.iloc[val_idx], y.iloc[val_idx]))

print(f"LogLoss promedio: {np.mean(scores):.3f}")
print(f"ROI promedio: {np.mean(rois)*100:.2f}%")

# Solo guarda si ROI > 2%
if np.mean(rois) > 0.02:
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    model.fit(X, y) # Entrena con todo
    joblib.dump(model, 'models/xg_model.pkl')
    df.to_csv('data/historico.csv', index=False)
    print("Modelo guardado. ROI > 2%")
else:
    print("Modelo descartado. ROI < 2%")
