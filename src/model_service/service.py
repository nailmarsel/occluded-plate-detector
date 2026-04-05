from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import logging
from embedding_model import CarEmbeddingModel
from database import EmbeddingDatabase

app = Flask(__name__)
CORS(app)

embedding_model = CarEmbeddingModel()
embedding_db = EmbeddingDatabase()

@app.route('/api/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Сохраняем временный файл
    fd, temp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    file.save(temp_path)
    
    try:
        # Генерируем эмбеддинг
        embedding = embedding_model.extract_embedding(temp_path)
        # Поиск похожих
        similar = embedding_db.get_similar_cars(embedding, top_k=3)
        # Возвращаем список ID
        car_ids = [item['car_id'] for item in similar]
        return jsonify(car_ids)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)