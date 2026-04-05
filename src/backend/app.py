from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests
import base64
from database import init_db, get_cars_by_ids, get_all_cars, add_car, get_car_by_id

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODEL_SERVICE_URL = 'http://localhost:5001/api/process'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_cars_with_images(car_ids):
    if not car_ids:
        return []
    cars = get_cars_by_ids(car_ids)
    for car in cars:
        photo_path = car['photo_path']
        if os.path.exists(photo_path):
            with open(photo_path, 'rb') as f:
                img_data = f.read()
                car['photo_base64'] = base64.b64encode(img_data).decode('utf-8')
        else:
            car['photo_base64'] = None
    return cars

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_car_image():
    if request.method == 'OPTIONS':
        return '', 200
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)
    try:
        with open(filename, 'rb') as f:
            files = {'file': f}
            response = requests.post(MODEL_SERVICE_URL, files=files)
        if response.status_code != 200:
            return jsonify({'error': 'Model service error'}), 500
        model_result = response.json()
        similar_car_ids = model_result if isinstance(model_result, list) else []
        matched_cars = get_cars_with_images(similar_car_ids)
        return jsonify({
            'success': True,
            'matched_cars': matched_cars,
            'car_ids': [c['id'] for c in matched_cars],
            'total_found': len(matched_cars)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filename):
            os.remove(filename)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'backend'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)