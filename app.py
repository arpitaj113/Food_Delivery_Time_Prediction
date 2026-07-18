from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model bundle
bundle = joblib.load("model.pkl")
model = bundle["model"]
mae = bundle["mae"]
rmse = bundle["rmse"]
r2 = bundle["r2"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        distance_km = float(request.form["distance_km"])
        weather = request.form["weather"]
        traffic_level = request.form["traffic_level"]
        time_of_day = request.form["time_of_day"]
        vehicle_type = request.form["vehicle_type"]
        preparation_time_min = float(request.form["preparation_time_min"])
        courier_experience_yrs = float(request.form["courier_experience_yrs"])

        input_data = pd.DataFrame({
            "distance_km": [distance_km],
            "weather": [weather],
            "traffic_level": [traffic_level],
            "time_of_day": [time_of_day],
            "vehicle_type": [vehicle_type],
            "preparation_time_min": [preparation_time_min],
            "courier_experience_yrs": [courier_experience_yrs]
        })

        prediction = model.predict(input_data)[0]

        return render_template(
            "result.html",
            prediction=round(prediction, 2),
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            r2=round(r2, 2)
        )

    except Exception as e:
        return render_template("result.html", error=str(e))

# Optional API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        input_data = pd.DataFrame({
            "distance_km": [float(data["distance_km"])],
            "weather": [data["weather"]],
            "traffic_level": [data["traffic_level"]],
            "time_of_day": [data["time_of_day"]],
            "vehicle_type": [data["vehicle_type"]],
            "preparation_time_min": [float(data["preparation_time_min"])],
            "courier_experience_yrs": [float(data["courier_experience_yrs"])]
        })

        prediction = model.predict(input_data)[0]

        return jsonify({
            "predicted_delivery_time_min": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)