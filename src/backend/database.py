import sqlite3
import os

DB_PATH = 'car_database.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT NOT NULL,
        photo_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

def add_car(plate_number, photo_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO cars (plate_number, photo_path) VALUES (?, ?)', 
                   (plate_number, photo_path))
    car_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return car_id

def get_car_by_id(car_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, plate_number, photo_path, created_at FROM cars WHERE id = ?', (car_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'plate_number': row[1], 'photo_path': row[2], 'created_at': row[3]}
    return None

def get_cars_by_ids(car_ids):
    if not car_ids:
        return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(car_ids))
    cursor.execute(f'SELECT id, plate_number, photo_path, created_at FROM cars WHERE id IN ({placeholders})', car_ids)
    rows = cursor.fetchall()
    conn.close()
    return [{'id': r[0], 'plate_number': r[1], 'photo_path': r[2], 'created_at': r[3]} for r in rows]

def get_all_cars():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, plate_number, photo_path, created_at FROM cars ORDER BY id')
    rows = cursor.fetchall()
    conn.close()
    return [{'id': r[0], 'plate_number': r[1], 'photo_path': r[2], 'created_at': r[3]} for r in rows]

def count_cars():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM cars')
    count = cursor.fetchone()[0]
    conn.close()
    return count