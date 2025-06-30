from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)
import pickle
# Load model and scalers
model = load_model('model.h5')
with open("scaler.pkl","rb") as f:
    scaler=pickle.load(f)
with open("labelEncoder.pkl","rb") as f:
    encoder=pickle.load(f)
    
df = pd.read_csv("Spotify.csv")


FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'tempo']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(feat)) for feat in FEATURES]
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)
    mood = encoder.inverse_transform([np.argmax(prediction)])[0]
    recommended = df[df["mood"] == mood][["track_name", "artists"]].sample(n=3, random_state=1).to_dict(orient="records")

    return render_template('index.html',
                            prediction_text=f'Predicted Mood: {mood}',
                            recommendations=recommended,
                            mood=mood)

if __name__ == '__main__':
    app.run(debug=True)
