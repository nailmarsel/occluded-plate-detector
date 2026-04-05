import sqlite3
import json
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class EmbeddingDatabase:
    def __init__(self, db_path='embeddings.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS car_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            car_id INTEGER NOT NULL UNIQUE,
            plate_number TEXT,
            embedding BLOB,
            photo_path TEXT,
            plate_visible_part TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()

    def add_embedding(self, car_id, plate_number, embedding, photo_path, plate_visible_part=None):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            embedding_json = json.dumps(embedding)
            c.execute('''
            INSERT OR REPLACE INTO car_embeddings 
            (car_id, plate_number, embedding, photo_path, plate_visible_part)
            VALUES (?, ?, ?, ?, ?)
            ''', (car_id, plate_number, embedding_json, photo_path, plate_visible_part))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"add_embedding error: {e}")
            return False

    def get_similar_cars(self, query_embedding, visible_part=None, top_k=3):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT car_id, plate_number, embedding, photo_path, plate_visible_part FROM car_embeddings')
        rows = c.fetchall()
        conn.close()

        query_np = np.array(query_embedding)
        results = []
        for row in rows:
            car_id, plate_number, emb_json, photo_path, plate_visible = row
            try:
                emb = np.array(json.loads(emb_json))
                similarity = np.dot(query_np, emb) / (np.linalg.norm(query_np) * np.linalg.norm(emb) + 1e-8)
                results.append({
                    'car_id': car_id,
                    'plate_number': plate_number,
                    'similarity': float(similarity),
                    'photo_path': photo_path,
                    'plate_visible_part': plate_visible
                })
            except:
                continue
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def count_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM car_embeddings')
        count = c.fetchone()[0]
        conn.close()
        return count

    def get_all_embeddings(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT car_id, plate_number, photo_path, plate_visible_part FROM car_embeddings')
        rows = c.fetchall()
        conn.close()
        return [{'car_id': r[0], 'plate_number': r[1], 'photo_path': r[2], 'plate_visible_part': r[3]} for r in rows]