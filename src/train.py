import pandas as pd, xgboost as xgb, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

df = pd.read_csv('data/partidos_historico.csv')
# Features: xG_diff, forma_ultimos5, dias_descanso, cuota_cierre, etc
X, y = df[features], df['resultado'] # 1=local, 0=empate, 2=visitante

tss = TimeSeriesSplit(n_splits=5)
modelo_nuevo = xgb.XGBClassifier()
scores = []
for train_idx, val_idx in tss.split(X):
    modelo_nuevo.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = modelo_nuevo.predict_proba(X.iloc[val_idx])
    scores.append(log_loss(y.iloc[val_idx], pred))

score_nuevo = sum(scores)/len(scores)

# Compara con modelo anterior
try:
    modelo_viejo = joblib.load('models/xg_model.pkl')
    score_viejo = # calculas igual
except: score_viejo = 99

if score_nuevo < score_viejo:
    joblib.dump(modelo_nuevo, 'models/xg_model.pkl')
    print("Nuevo modelo es mejor. Guardado.")
else:
    print("Modelo anterior sigue ganando. No se actualiza.")
