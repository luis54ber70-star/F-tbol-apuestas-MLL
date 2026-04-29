import pandas as pd, requests, joblib, os, numpy as np
from datetime import datetime

API_KEY = os.getenv('API_KEY')
LEAGUE_ID = 262
MODEL_PATH = 'models/xg_model.pkl'
DATA_PATH = 'data/historico.csv'

os.makedirs('predictions', exist_ok=True)

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    with open('predictions/hoy.md', 'w') as f:
        f.write("# Sin modelo entrenado\nCorre el workflow Train primero.")
    print("No hay modelo o datos. Corre Train primero.")
    exit()

model = joblib.load(MODEL_PATH)
df_hist = pd.read_csv(DATA_PATH)

# Traer fixtures de hoy
today = datetime.utcnow().strftime('%Y-%m-%d')
url = f"https://v3.football.api-sports.io/fixtures?league={LEAGUE_ID}&season=2024&date={today}"
r = requests.get(url, headers={"x-apisports-key": API_KEY})
r.raise_for_status()
fixtures = r.json()['response']

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
    prob = model.predict_proba(X)[0] # [Empate, Local, Visitante]

    # Kelly 25% asumiendo cuota 2.0
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
    f.write("Modelo: XGBoost | Kelly 25% | Umbral 52%\n\n")
    if picks:
        for p in picks: f.write(f"- {p}\n")
    else:
        f.write("Sin valor detectado hoy.\n")

print("Predicciones generadas")
