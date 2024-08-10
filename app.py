from flask import request, Flask, jsonify
import logging
from scripts.predict import pred

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)


@app.route('/predict', methods=['POST'])
def predict():
    # try:
    data = request.get_json()

    # Check for required fields in the request
    if not all(key in data for key in ('date', 'crop', 'region')):
        return jsonify({"error": "Missing required fields"}), 400

    date_str = data['date']
    crop = data['crop']
    region = data['region']

    predictions = pred(date_str, crop, region)

    return jsonify({
        'predictions': predictions,
    })



if __name__ == "__main__":
    app.run(debug=True)
