F-tbol-apuestas-MLL

Bot de predicciones Liga MX usando XGBoost + Kelly 25%. Corre automático con GitHub Actions y te da picks diarios en `predictions/hoy.md`.

**Cómo funciona**

1. **Train**: Descarga históricos de Liga MX, entrena XGBoost y guarda `models/xg_model.pkl`
2. **Predict**: Cada día revisa partidos de Liga MX, calcula probabilidades y sugiere apuestas solo si `prob > 52%`
3. **Kelly 25%**: Calcula stake = `EV / cuota * 0.25` para gestionar bankroll

**Setup en 3 minutos**

**1. Fork y Secret**

1. Fork este repo
2. Ve a `Settings → Secrets and variables → Actions → New repository secret`
3. Name: `API_KEY`
4. Secret: Tu key de [API-Football](https://www.api-football.com/)

**2. Estructura del repo**

F-tbol-apuestas-MLL/
├──.github/workflows/
│ ├── http://predict.yml # Corre 8pm y 11pm CDMX
│ └── http://train.yml # Corre 9am CDMX
├── src/
│ ├── http://train.py # Entrena modelo
│ └── http://predict.py # Genera picks del día
├── models/ # xg_model.pkl se crea solo
├── data/ # http://historico.csv se crea solo
├── predictions/ # http://hoy.md se crea solo
└── http://requirements.txt

**3. Primera corrida**

1. Ve a `Actions → Auto-Train → Run workflow` para crear el modelo
2. Ve a `Actions → Auto-Predict Liga MX → Run workflow` para ver picks de hoy
3. Revisa `predictions/hoy.md` en el repo

**Archivos clave**

**`requirements.txt`**
pandas
requests
joblib
xgboost
scikit-learn

**`.github/workflows/train.yml`**
```yaml
name: Auto-Train
on:
  schedule:
    - cron: '0 15 * * *' # 9:00am CDMX
  workflow_dispatch:
jobs:
  train:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with: {lfs: true}
      - uses: actions/setup-python@v5
        with: {python-version: '3.12', cache: 'pip'}
      - run: pip install -r requirements.txt
      - name: Train model
        env:
          API_KEY: ${{ secrets.API_KEY }}
        run: python src/train.py
      - name: Commit model
        run: |
          git config user.name github-actions
          git config user.email actions@github.com
          mkdir -p models data
          git add models/ data/ || true
          git diff --staged --quiet || git commit -m "Auto-retrain $(TZ=America/Mexico_City date +'%Y-%m-%d')"
          git push || echo "No changes"
*`.github/workflows/predict.yml`*
name: Auto-Predict Liga MX
on:
  schedule:
    - cron: '0 2,5 * * *' # 8pm y 11pm CDMX
  workflow_dispatch:
jobs:
  predict:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.12', cache: 'pip'}
      - run: pip install -r requirements.txt
      - name: Run predictions
        env:
          API_KEY: ${{ secrets.API_KEY }}
        run: python src/predict.py
      - name: Commit predictions
        run: |
          git config user.name github-actions
          git config user.email actions@github.com
          git add predictions/ || true
          git diff --staged --quiet || git commit -m "Picks auto $(TZ=America/Mexico_City date +'%Y-%m-%d %H:%M')"
          git push || echo "No changes"
*`src/train.py`*
import pandas as pd, requests, joblib, os, numpy as np, sys
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

API_KEY = os.getenv('API_KEY')
LEAGUE_ID = 262 # Liga MX
TRAIN_SEASONS = # 2024=Apertura 2024, 2026=Clausura 2026

if not API_KEY:
    print("ERROR: No se encontró API_KEY en Secrets.")
    sys.exit(1)

def get_data():
    all_rows = []
    for season in TRAIN_SEASONS:
        url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={season}"
        print(f"Consultando season {season}: {url}")
        r = requests.get(url, headers={"x-apisports-key": API_KEY})
        r.raise_for_status()
        data = r.json()
        if 'response' not in data:
            print(f"ERROR API season {season}: {data}")
            continue
        fixtures = data['response']
        print(f"Fixtures recibidos de API season {season}: {len(fixtures)}")
        for f in fixtures:
            if f['fixture']['status']['short']!= 'FT': continue
            h = f['teams']['home']['name']
            a = f['teams']['away']['name']
            gh, ga = f['goals']['home'], f['goals']['away']
            if gh is None or ga is None: continue
            if gh > ga: result = 1
            elif gh == ga: result = 0
            else: result = 2
            all_rows.append({'date': f['fixture']['date'], 'home': h, 'away': a,
                         'gh': gh, 'ga': ga, 'result': result})
    if len(all_rows) == 0:
        print(f"ERROR: La API no regresó partidos finalizados para seasons {TRAIN_SEASONS}.")
        sys.exit(1)
    df = pd.DataFrame(all_rows).sort_values('date')
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

model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
model.fit(X, y)
joblib.dump(model, 'models/xg_model.pkl')
df.to_csv('data/historico.csv', index=False)
print("Modelo guardado con seasons 2024 + 2026.")
*`src/predict.py`*
import pandas as pd, requests, joblib, os, numpy as np, sys
from datetime import datetime
from zoneinfo import ZoneInfo

API_KEY = os.getenv('API_KEY')
LEAGUE_ID = 262
PREDICT_SEASON = 2026 # Clausura 2026
MODEL_PATH = 'models/xg_model.pkl'
DATA_PATH = 'data/historico.csv'

os.makedirs('predictions', exist_ok=True)
today = datetime.now(ZoneInfo("America/Mexico_City")).strftime('%Y-%m-%d')

if not API_KEY:
    with open('predictions/hoy.md', 'w') as f:
        f.write(f"# Error\nNo se encontró API_KEY en Secrets.\n")
    sys.exit(1)

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    with open('predictions/hoy.md', 'w') as f:
        f.write(f"# Picks Liga MX - {today}\n\n")
        f.write("Sin modelo entrenado. Corre el workflow Train primero.\n")
    sys.exit(0)

model = joblib.load(MODEL_PATH)
df_hist = pd.read_csv(DATA_PATH)

url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season={PREDICT_SEASON}&date={today}"
print(f"Consultando fixtures de hoy: {url}")
r = requests.get(url, headers={"x-apisports-key": API_KEY})
r.raise_for_status()
data = r.json()
fixtures = data.get('response', [])
print(f"Partidos encontrados para hoy: {len(fixtures)}")

if len(fixtures) == 0:
    with open('predictions/hoy.md', 'w') as f:
        f.write(f"# Picks Liga MX - {today}\n\n")
        f.write("No hay partidos de Liga MX programados para hoy.\n")
        f.write(f"\nModelo entrenado con {len(df_hist)} partidos históricos.\n")
    print("Sin partidos hoy")
    sys.exit(0)

def get_team_stats(team, is_home):
    col_gf = 'gh' if is_home else 'ga'
    col_ga = 'ga' if is_home else 'gh'
    team_col = 'home' if is_home else 'away'
    last5 = df_hist[df_hist[team_col] == team].tail(5)
    if len(last5) == 0: return 1.2, 1.2
    return last5[col_gf].mean(), last5[col_ga].mean()

picks = []
for f in fixtures:
    if f['fixture']['status']['short']!= 'NS': continue
    home, away = f['teams']['home']['name'], f['teams']['away']['name']
    h_gf, h_ga = get_team_stats(home, True)
    a_gf, a_ga = get_team_stats(away, False)
    goal_diff_avg = (h_gf - h_ga) - (a_gf - a_ga)
    X = pd.DataFrame([[goal_diff_avg, h_gf, h_ga, a_gf, a_ga]],
                     columns=['goal_diff_avg','home_gf_avg','home_ga_avg','away_gf_avg','away_ga_avg'])
    prob = model.predict_proba(X)[0]
    ev_local = prob[1] * 2.0 - 1
    ev_visit = prob[2] * 2.0 - 1
    kelly_local = max(0, ev_local / 1.0) * 0.25
    kelly_visit = max(0, ev_visit / 1.0) * 0.25
    if prob[1] > 0.52:
        picks.append(f"**{home} vs {away}** | Local {prob[1]*100:.1f}% | Kelly: {kelly_local*100:.1f}% bankroll")
    elif prob[2] > 0.52:
        picks.append(f"**{home} vs {away}** | Visitante {prob[2]*100:.1f}% | Kelly: {kelly_visit*100:.1f}% bankroll")

with open('predictions/hoy.md', 'w') as f:
    f.write(f"# Picks Liga MX - {today}\n\n")
    f.write("Modelo: XGBoost entrenado con 2024+2026 | Kelly 25% | Umbral 52%\n\n")
    if picks:
        for p in picks: f.write(f"- {p}\n")
    else:
        f.write("Sin valor detectado hoy.\n")
    f.write(f"\n*Datos históricos: {len(df_hist)} partidos*\n")
print("Predicciones generadas")
*Lógica del modelo*
Componente	Detalle
**Datos**	Liga MX 2024 + 2026, solo partidos FT
**Features**	Promedio goles favor/contra últimos 5 partidos, goal_diff_avg
**Modelo**	XGBClassifier: 200 árboles, max_depth=3
**Validación**	TimeSeriesSplit 3 folds + backtest ROI
**Filtro picks**	Solo si probabilidad > 52%
**Stake**	Kelly 25%: `max(0, EV/1.0) * 0.25`
*Ejemplo de output `predictions/hoy.md`*
Picks Liga MX - 2026-05-02

Modelo: XGBoost entrenado con 2024+2026 | Kelly 25% | Umbral 52%

- **Tigres vs Chivas** | Local 56.3% | Kelly: 3.2% bankroll
- **Atlas vs Cruz Azul** | Visitante 53.1% | Kelly: 1.4% bankroll

*Datos históricos: 518 partidos*
*Notas importantes*

1. *Horarios*: Todos los cron están en UTC. `02:00 UTC = 8pm CDMX`, `15:00 UTC = 9am CDMX`
2. *Temporada*: Usa `TRAIN_SEASONS = ` en `train.py`. Cambia a `` cuando acabe Clausura
3. *Límite API*: API-Football gratis da 100 requests/día. El bot usa ∼3-4 por corrida
4. *Sin valor*: Si no hay picks, es porque ninguna prob > 52%. No forzar apuestas
5. *Bankroll*: Kelly 25% es agresivo. Si quieres menos riesgo cambia `_ 0.25` por `_ 0.10` en `predict.py`

*Troubleshooting*
Error	Fix
`No se encontró API_KEY`	Agrega el secret en Settings → Secrets → Actions
`Fixtures recibidos: 0`	Temporada 2026 aún sin FT. Usa `[2024]` o espera
`Sin modelo entrenado`	Corre workflow Train primero
`No hay partidos hoy`	Normal. Liga MX no juega diario
*Disclaimer*

Esto es un modelo estadístico, no garantía de ganancia. Apuesta responsable. El deporte tiene varianza
