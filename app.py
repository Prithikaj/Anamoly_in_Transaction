from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

model_path = 'models/isolation_forest_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')  # Serve HTML form

@app.route('/predict', methods=['POST'])
def predict():
    transaction = {
        "Transaction ID": request.form['Transaction ID'],
        "Date": request.form['Date'],
        "Account Number": request.form['Account Number'],
        "Amount": request.form['Amount'],
        "Transaction Type": request.form['Transaction Type'],
        "Description": request.form['Description']
    }

    df = pd.DataFrame([transaction])
    df['Date'] = pd.to_datetime(df['Date'])
    X_new = df[['Amount']]
    prediction = model.predict(X_new)
    is_anomaly = 1 if prediction[0] == -1 else 0
    return jsonify({'Anomaly': is_anomaly})

if __name__ == '__main__':
    app.run(debug=True)
