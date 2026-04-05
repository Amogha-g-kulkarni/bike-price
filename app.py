from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
from flask import send_file
import os




# Create app
app = Flask(__name__)
CORS(app)

# Load model (runs once)
model = joblib.load('model/bike_price_model.pkl')


# Health check (optional but important)
@app.route('/')
def home():
    return jsonify({
        "status": "running",
        "message": "Bike Price API is live 🚀"
    })

@app.route('/ui')
def ui():
    return send_file('index.html')


# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Step 1: Get JSON data
        data = request.get_json()

        # Step 2: Convert to DataFrame
        df = pd.DataFrame([data])

        required_fields = ["Age", "km_driven", "ex_showroom_price","brand", "model", "seller_type", "owner"]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is missing"})
        
        # try:
        #     data["Age"] = int(data["Age"])
        #     data["km_driven"] = int(data["km_driven"])
        #     data["ex_showroom_price"] = float(data["ex_showroom_price"])
        # except:
        #     return jsonify({"error": "Invalid numeric values"})

        # Step 3: Prediction (log scale)
        pred_log = model.predict(df)

        # Step 4: Convert back
        price = np.expm1(pred_log)

        # Step 5: Return response
        return jsonify({
            "status": "success",
            "predicted_price": float(price[0])})

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)})


# Run server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)