from flask import request, Flask, jsonify
import logging
from scripts.predict import pred

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check for required fields in the request
        if not all(key in data for key in ('date', 'crop', 'region', 'temp', 'humi', 'monthly_rainfall_mm')):
            return jsonify({"error": "Missing required fields"}), 400

        date_str = data['date']
        crop = data['crop']
        region = data['region']
        temp = data['temp']
        humi = data['humi']
        rainfall = data['monthly_rainfall_mm']

        predictions, evaluation_results = pred(date_str, crop, region, temp, humi, rainfall)

        return jsonify({
            'predictions': predictions,
            'evaluation_results': evaluation_results
        })

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

