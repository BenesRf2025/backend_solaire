# app.py (Complet - Accepte GET & POST)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app) # Autorise Flutter à se connecter

# CONFIGURATION MÉTÉO (COMPTE DEVELOPER)
OPENWEATHER_API_KEY = "MON_API_KEY_ICI"
CITY_LAT = "-23.35"  # Toliara
CITY_LON = "43.66"

# --- 1. ROUTE D'ACCUEIL ---
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "status": "online",
        "system": "Solar AI ",
        "location": "Toliara",
        "weather_server": "pro.openweathermap.org"
    })

# --- 2. FONCTION MÉTÉO ---
def get_weather_data():
    try:
        url = f"https://pro.openweathermap.org/data/2.5/weather?lat={CITY_LAT}&lon={CITY_LON}&appid={OPENWEATHER_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            clouds = data['clouds']['all']
            temp = data['main']['temp'] - 273.15  # Convertir Kelvin en Celsius
            return clouds, temp
        else:
            print(f"Erreur Météo : {response.status_code}")
    except Exception as e:
        print(f"Exception Météo: {e}")
    return 0, 25  # Par défaut: beau temps, 25°C

# --- 3. ENTRAÎNEMENT DU MODÈLE AVEC VALIDATION ---
def train_initial_model():
    print("Entraînement du modèle IA avec validation croisée...")
    data = []
    for _ in range(3000):
        hour = np.random.randint(0, 24)
        battery = np.random.uniform(11.0, 14.5)
        solar = 0
        conso = np.random.uniform(20, 150)
        clouds = np.random.randint(0, 100)
        temp = np.random.uniform(20, 35)  # Température ajoutée
        season = np.random.choice([0, 1, 2, 3])  # Saison (0=hiver, 1=printemps, etc.)

        if 7 <= hour <= 17:
            # Ajustement saisonnier de la production solaire
            seasonal_factor = 1.0
            if season == 0: seasonal_factor = 0.7  # Hiver
            elif season == 3: seasonal_factor = 0.8  # Automne
            solar_potential = 300 * seasonal_factor * (1 - (clouds / 100))
            solar = max(0, np.random.normal(solar_potential, 20))

        should_turn_on = 0
        if battery > 12.8: should_turn_on = 1
        elif battery > 12.0 and clouds < 20 and 9 <= hour <= 15: should_turn_on = 1
        elif battery < 12.2 and clouds > 60: should_turn_on = 0
        elif solar > (conso + 10): should_turn_on = 1

        data.append([hour, battery, solar, conso, clouds, temp, season, should_turn_on])

    df = pd.DataFrame(data, columns=['hour', 'battery', 'solar', 'conso', 'clouds', 'temp', 'season', 'target'])

    # Validation croisée simple
    from sklearn.model_selection import train_test_split
    X = df[['hour', 'battery', 'solar', 'conso', 'clouds', 'temp', 'season']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Évaluation
    accuracy = model.score(X_test, y_test)
    print(f"Précision du modèle: {accuracy:.2%}")

    joblib.dump(model, 'solar_model_pro.pkl')
    return model

if os.path.exists('solar_model_pro.pkl'):
    model = joblib.load('solar_model_pro.pkl')
else:
    model = train_initial_model()

# --- 3.5. LOGGING DES PRÉDICTIONS ---
def log_prediction(battery, solar, conso, clouds, temp, season, decision):
    try:
        import csv
        import os
        file_exists = os.path.isfile('predictions_log.csv')
        with open('predictions_log.csv', 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'battery', 'solar', 'conso', 'clouds', 'temp', 'season', 'decision']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'battery': battery,
                'solar': solar,
                'conso': conso,
                'clouds': clouds,
                'temp': temp,
                'season': season,
                'decision': int(decision)
            })
    except Exception as e:
        print(f"Erreur logging: {e}")

# --- 4. API DE PRÉDICTION (GET/POST) ---
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Gestion intelligente GET vs POST
        if request.method == 'POST':
            data = request.json
            if not data: return jsonify({"error": "No JSON data"}), 400
            battery = float(data.get('battery', 0))
            solar = float(data.get('solar', 0))
            conso = float(data.get('conso', 0))
        else: # GET (Navigateur ou erreur Flutter)
            battery = float(request.args.get('battery', 0))
            solar = float(request.args.get('solar', 0))
            conso = float(request.args.get('conso', 0))

        current_hour = datetime.now().hour
        current_clouds, current_temp = get_weather_data()

        # Calcul de la saison (0=hiver, 1=printemps, 2=été, 3=automne)
        current_month = datetime.now().month
        if current_month in [12, 1, 2]: season = 0  # Hiver
        elif current_month in [3, 4, 5]: season = 1  # Printemps
        elif current_month in [6, 7, 8]: season = 2  # Été
        else: season = 3  # Automne

        features_df = pd.DataFrame([[current_hour, battery, solar, conso, current_clouds, current_temp, season]],
                                   columns=['hour', 'battery', 'solar', 'conso', 'clouds', 'temp', 'season'])
        prediction = model.predict(features_df)[0]
        decision = bool(prediction == 1)

        reason = "Conditions normales"
        if decision:
            if battery > 12.8: reason = "Batterie pleine"
            elif current_clouds < 30: reason = "Ciel dégagé, recharge rapide"
            else: reason = "Production suffisante"
        else:
            if battery < 11.5: reason = "Batterie critique"
            elif current_clouds > 70: reason = "Ciel trop couvert"
            else: reason = "Délestage préventif IA"

        print(f"IA Decision: {'ON' if decision else 'OFF'} | Raison: {reason}")

        # Log des prédictions pour réentraînement futur
        log_prediction(battery, solar, conso, current_clouds, current_temp, season, decision)

        return jsonify({
            "should_turn_on": decision,
            "weather_used": f"{current_clouds}%, {current_temp:.1f}°C",
            "season": ["Hiver", "Printemps", "Été", "Automne"][season],
            "reason": reason
        })

    except Exception as e:
        print(f"Erreur: {e}")
        return jsonify({"error": str(e)}), 500

# --- 5. ENDPOINT DE RÉENTRAÎNEMENT ---
@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        if not os.path.exists('predictions_log.csv'):
            return jsonify({"error": "Aucune donnée de log disponible"}), 400

        # Charger les données de log
        df_log = pd.read_csv('predictions_log.csv')

        # Combiner avec données synthétiques pour plus de robustesse
        synthetic_data = []
        for _ in range(1000):  # Moins que l'entraînement initial
            hour = np.random.randint(0, 24)
            battery = np.random.uniform(11.0, 14.5)
            solar = 0
            conso = np.random.uniform(20, 150)
            clouds = np.random.randint(0, 100)
            temp = np.random.uniform(20, 35)
            season = np.random.choice([0, 1, 2, 3])

            if 7 <= hour <= 17:
                seasonal_factor = 1.0
                if season == 0: seasonal_factor = 0.7
                elif season == 3: seasonal_factor = 0.8
                solar_potential = 300 * seasonal_factor * (1 - (clouds / 100))
                solar = max(0, np.random.normal(solar_potential, 20))

            should_turn_on = 0
            if battery > 12.8: should_turn_on = 1
            elif battery > 12.0 and clouds < 20 and 9 <= hour <= 15: should_turn_on = 1
            elif battery < 12.2 and clouds > 60: should_turn_on = 0
            elif solar > (conso + 10): should_turn_on = 1

            synthetic_data.append([hour, battery, solar, conso, clouds, temp, season, should_turn_on])

        df_synthetic = pd.DataFrame(synthetic_data, columns=['hour', 'battery', 'solar', 'conso', 'clouds', 'temp', 'season', 'target'])

        # Combiner les données
        df_combined = pd.concat([df_log[['battery', 'solar', 'conso', 'clouds', 'temp', 'season', 'decision']].rename(columns={'decision': 'target'}),
                                df_synthetic], ignore_index=True)

        # Réentraîner le modèle
        X = df_combined[['hour', 'battery', 'solar', 'conso', 'clouds', 'temp', 'season']]
        y = df_combined['target']
        new_model = RandomForestClassifier(n_estimators=100, random_state=42)
        new_model.fit(X, y)

        # Sauvegarder
        joblib.dump(new_model, 'solar_model_pro.pkl')

        # Recharger le modèle global
        global model
        model = new_model

        return jsonify({
            "status": "success",
            "message": f"Modèle réentraîné avec {len(df_combined)} échantillons",
            "accuracy_estimate": f"{new_model.score(X, y):.2%}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)