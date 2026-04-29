import pandas as pd, requests, joblib, os, numpy as np, sys
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

API_KEY = os.getenv('API_KEY')
LEAGUE_ID = 262 # Liga MX
SEASON = 2025 # Apertura 2025 + Clausura 2026

if not API_KEY:
    print("ERROR: No se encontró API_KEY. Configura API_FOOTBALL_KEY en Secrets.")
    sys.exit(1)

def get_data():
    url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={SEASON}"
    print(f"Consultando: {url}")
    r = requests.get(url, headers={"x-apisports-key": API_KEY})
    r.raise_for_status()
    data = r.json()
    
    if 'response' not in data:
        print(f"ERROR API: {data}")
        sys.exit(1)
    
    fixtures = data['response']
    print(f"Fixtures recibidos de API: {len(fixtures)}")

    rows = []
    for f in fixtures:
        if f['fixture']['status']['short']!= 'FT': continue
        h = f['teams']['home']['name']
        a = f['teams']['away']['name']
        gh, ga = f['goals']['home'], f['goals']['away']
        if gh is None or ga is None: continue
        if gh > ga: result = 1
        elif gh == ga: result = 0
        else: result = 2
        rows.append({'date': f['fixture']['date'], 'home': h, 'away': a,
                     'gh': gh, 'ga': ga, 'result': result})

    if len(rows) == 0:
        print("ERROR: La API no regresó partidos finalizados para season 2025.")
        print("Posibles causas:")
        print("1. API key sin permisos o límite alcanzado")
        print("2. La season 2025 aún no tiene partidos en la API")
        print("3. Cambia SEASON = 2024 para probar con datos del año pasado")
        sys.exit(1)

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
    wins = (picks == y).sum()
    bets = mask.sum()
    roi = (wins * 0.95 - (bets - wins)) / bets
    return roi

df = get_data()
print(f"Partidos históricos encontrados: {len(df)}")
if len(df) < 50:
    print("No hay suficientes datos históricos. Se necesitan 50+ partidos.")
    sys.exit(1)

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

if np.mean(rois) > 0.02 or not os.path.exists('models/xg_model.pkl'):
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    model.fit(X, y)
    joblib.dump(model, 'models/xg_model.pkl')
    df.to_csv('data/historico.csv', index=False)
    print("Modelo guardado.")
else:
    print("Modelo descartado. ROI < 2%")
