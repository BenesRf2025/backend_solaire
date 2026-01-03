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

# 🌤️ CONFIGURATION MÉTÉO (COMPTE DEVELOPER)
OPENWEATHER_API_KEY = "7e29c591e985d0bbe66a5fcc395aed2f"
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
def get_weather_cloudiness():
    try:
        url = f"https://pro.openweathermap.org/data/2.5/weather?lat={CITY_LAT}&lon={CITY_LON}&appid={OPENWEATHER_API_KEY}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['clouds']['all']
        else:
            print(f"Erreur Météo : {response.status_code}")
    except Exception as e:
        print(f"Exception Météo: {e}")
    return 0 # Par défaut: beau temps

# --- 3. ENTRAÎNEMENT DU MODÈLE ---
def train_initial_model():
    print("Entraînement du modèle IA...")
    data = []
    for _ in range(3000):
        hour = np.random.randint(0, 24)
        battery = np.random.uniform(11.0, 14.5)
        solar = 0
        conso = np.random.uniform(20, 150)
        clouds = np.random.randint(0, 100)
        
        if 7 <= hour <= 17:
            solar_potential = 300 * (1 - (clouds / 100))
            solar = max(0, np.random.normal(solar_potential, 20))
            
        should_turn_on = 0
        if battery > 12.8: should_turn_on = 1
        elif battery > 12.0 and clouds < 20 and 9 <= hour <= 15: should_turn_on = 1
        elif battery < 12.2 and clouds > 60: should_turn_on = 0
        elif solar > (conso + 10): should_turn_on = 1
            
        data.append([hour, battery, solar, conso, clouds, should_turn_on])

    df = pd.DataFrame(data, columns=['hour', 'battery', 'solar', 'conso', 'clouds', 'target'])
    model = RandomForestClassifier(n_estimators=100)
    model.fit(df[['hour', 'battery', 'solar', 'conso', 'clouds']], df['target'])
    joblib.dump(model, 'solar_model_pro.pkl')
    return model

if os.path.exists('solar_model_pro.pkl'):
    model = joblib.load('solar_model_pro.pkl')
else:
    model = train_initial_model()

# --- 4. API DE PRÉDICTION (CORRIGÉE GET/POST) ---
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
        current_clouds = get_weather_cloudiness()
        
        features = [[current_hour, battery, solar, conso, current_clouds]]
        prediction = model.predict(features)[0]
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

        return jsonify({
            "should_turn_on": decision,
            "weather_used": f"{current_clouds}%",
            "reason": reason
        })

    except Exception as e:
        print(f"Erreur: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
