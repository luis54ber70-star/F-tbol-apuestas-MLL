# ⚽ Fútbol Apuestas MLL (Machine Learning Liga MX)

Este repositorio contiene un sistema automatizado de Machine Learning diseñado para predecir los resultados de los partidos de la **Liga MX** (Fútbol Mexicano). Utiliza datos históricos y en tiempo real, entrenando un modelo para calcular las probabilidades de victoria local, empate o victoria visitante.

## 🌟 Características Principales

* **Extracción Automática de Datos:** Se conecta a [API-Sports](https://www.api-football.com/) para descargar resultados históricos y los partidos programados del día.
* **Feature Engineering:** Calcula promedios móviles de goles a favor (GF), goles en contra (GC) y diferenciales de goles para medir el rendimiento reciente de los equipos.
* **Modelo Predictivo:** Utiliza `XGBClassifier` (XGBoost) optimizado para evaluar las probabilidades de los partidos.
* **Backtesting de ROI:** Incluye un sistema de simulación de apuestas basado en la confianza del modelo (probabilidades > 52%) para calcular el Retorno de Inversión (ROI) histórico.
* **Automatización CI/CD:** Preparado para ejecutarse mediante GitHub Actions.

---

## 📂 Estructura del Proyecto

```text
F-tbol-apuestas-MLL/
│
├── src/
│   ├── train.py       # Descarga datos, crea features, entrena y guarda el modelo.
│   └── predict.py     # Carga el modelo y predice los partidos del día actual.
│
├── models/            # Directorio generado donde se guarda el modelo (xg_model.pkl).
├── data/              # Directorio generado para guardar el dataset histórico (historico.csv).
├── requirements.txt   # Dependencias de Python necesarias.
└── README.md          # Documentación del proyecto.
