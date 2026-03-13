import sqlite3
from datetime import datetime, timedelta
import json
import os

DB_PATH = 'lettuce_farm.db'

def init_database():
    """Initialize database with all tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sensor readings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            temperature REAL,
            humidity REAL,
            sensor_id TEXT,
            location TEXT
        )
    ''')
    
    # Detections table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            label TEXT,
            confidence REAL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            health_status TEXT,
            health_confidence REAL,
            image_path TEXT
        )
    ''')
    
    # Relay events table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS relay_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            relay_name TEXT,
            action TEXT,
            trigger_type TEXT,
            temperature REAL,
            humidity REAL
        )
    ''')
    
    # Daily summary table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE UNIQUE,
            total_detections INTEGER,
            ready_count INTEGER,
            not_ready_count INTEGER,
            avg_temperature REAL,
            avg_humidity REAL,
            peak_ready_count INTEGER,
            min_temperature REAL,
            max_temperature REAL,
            min_humidity REAL,
            max_humidity REAL
        )
    ''')
    
    # System logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            log_level TEXT,
            component TEXT,
            message TEXT,
            details TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")
    print(f"📁 Database location: {os.path.abspath(DB_PATH)}")

# ========================================
# SENSOR READING FUNCTIONS
# ========================================

def save_sensor_reading(temperature, humidity, sensor_id='DHT11_001', location='Greenhouse_A'):
    """Save sensor reading to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_readings (temperature, humidity, sensor_id, location)
            VALUES (?, ?, ?, ?)
        ''', (temperature, humidity, sensor_id, location))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error saving sensor reading: {e}")
        return False

def get_sensor_history(hours=24):
    """Get sensor readings from last N hours"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, temperature, humidity, sensor_id, location
            FROM sensor_readings
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp ASC
        '''.format(hours))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'timestamp': row[0],
            'temperature': row[1],
            'humidity': row[2],
            'sensor_id': row[3],
            'location': row[4]
        } for row in rows]
    except Exception as e:
        print(f"❌ Error getting sensor history: {e}")
        return []

def get_latest_sensor_reading():
    """Get the most recent sensor reading"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, temperature, humidity, sensor_id, location
            FROM sensor_readings
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'timestamp': row[0],
                'temperature': row[1],
                'humidity': row[2],
                'sensor_id': row[3],
                'location': row[4]
            }
        return None
    except Exception as e:
        print(f"❌ Error getting latest reading: {e}")
        return None

# ========================================
# DETECTION FUNCTIONS
# ========================================

def save_detection(label, confidence, bbox, health_status='Unknown', health_confidence=0.0, image_path=''):
    """Save detection to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections 
            (label, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
             health_status, health_confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (label, confidence, bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
              health_status, health_confidence, image_path))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error saving detection: {e}")
        return False

def get_detection_summary(days=30):
    """Get detection summary grouped by date and label"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                label,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM detections
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY DATE(timestamp), label
            ORDER BY date DESC
        '''.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'date': row[0],
            'label': row[1],
            'count': row[2],
            'avg_confidence': row[3]
        } for row in rows]
    except Exception as e:
        print(f"❌ Error getting detection summary: {e}")
        return []

def get_detections_by_date(date):
    """Get all detections for a specific date"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, label, confidence, 
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                   health_status, health_confidence, image_path
            FROM detections
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        ''', (date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'timestamp': row[1],
            'label': row[2],
            'confidence': row[3],
            'bbox': {
                'x1': row[4], 'y1': row[5],
                'x2': row[6], 'y2': row[7]
            },
            'health_status': row[8],
            'health_confidence': row[9],
            'image_path': row[10]
        } for row in rows]
    except Exception as e:
        print(f"❌ Error getting detections by date: {e}")
        return []

# ========================================
# RELAY EVENT FUNCTIONS
# ========================================

def save_relay_event(relay_name, action, trigger_type, temperature=None, humidity=None):
    """Save relay event to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO relay_events 
            (relay_name, action, trigger_type, temperature, humidity)
            VALUES (?, ?, ?, ?, ?)
        ''', (relay_name, action, trigger_type, temperature, humidity))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error saving relay event: {e}")
        return False

def get_relay_history(days=7):
    """Get relay events from last N days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, relay_name, action, trigger_type, temperature, humidity
            FROM relay_events
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'timestamp': row[0],
            'relay': row[1],
            'action': row[2],
            'trigger': row[3],
            'temperature': row[4],
            'humidity': row[5]
        } for row in rows]
    except Exception as e:
        print(f"❌ Error getting relay history: {e}")
        return []

def get_relay_stats(days=7):
    """Get relay usage statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                relay_name,
                action,
                COUNT(*) as count,
                trigger_type
            FROM relay_events
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY relay_name, action, trigger_type
        '''.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'relay': row[0],
            'action': row[1],
            'count': row[2],
            'trigger': row[3]
        } for row in rows]
    except Exception as e:
        print(f"❌ Error getting relay stats: {e}")
        return []

# ========================================
# DAILY SUMMARY FUNCTIONS
# ========================================

def update_daily_summary(date=None):
    """Update daily summary for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get detection counts
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN label LIKE '%Ready%' AND label NOT LIKE '%Not%' THEN 1 ELSE 0 END) as ready,
                SUM(CASE WHEN label LIKE '%Not Ready%' OR label LIKE '%small size%' THEN 1 ELSE 0 END) as not_ready,
                MAX(CASE WHEN label LIKE '%Ready%' AND label NOT LIKE '%Not%' THEN 1 ELSE 0 END) as peak_ready
            FROM detections
            WHERE DATE(timestamp) = ?
        ''', (date,))
        
        det_row = cursor.fetchone()
        
        # Get sensor averages
        cursor.execute('''
            SELECT 
                AVG(temperature) as avg_temp,
                AVG(humidity) as avg_humid,
                MIN(temperature) as min_temp,
                MAX(temperature) as max_temp,
                MIN(humidity) as min_humid,
                MAX(humidity) as max_humid
            FROM sensor_readings
            WHERE DATE(timestamp) = ?
        ''', (date,))
        
        sensor_row = cursor.fetchone()
        
        # Insert or update summary
        cursor.execute('''
            INSERT OR REPLACE INTO daily_summary 
            (date, total_detections, ready_count, not_ready_count, 
             avg_temperature, avg_humidity, peak_ready_count,
             min_temperature, max_temperature, min_humidity, max_humidity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, 
              det_row[0] or 0, det_row[1] or 0, det_row[2] or 0,
              sensor_row[0], sensor_row[1], det_row[3] or 0,
              sensor_row[2], sensor_row[3], sensor_row[4], sensor_row[5]))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error updating daily summary: {e}")
        return False

def get_daily_summaries(days=30):
    """Get daily summaries for last N days"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, total_detections, ready_count, not_ready_count,
                   avg_temperature, avg_humidity, peak_ready_count,
                   min_temperature, max_temperature, min_humidity, max_humidity
            FROM daily_summary
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
        '''.format(days))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'date': row[0],
            'total_detections': row[1],
            'ready_count': row[2],
            'not_ready_count': row[3],
            'avg_temperature': row[4],
            'avg_humidity': row[5],
            'peak_ready_count': row[6],
            'min_temperature': row[7],
            'max_temperature': row[8],
            'min_humidity': row[9],
            'max_humidity': row[10]
        } for row in rows]
    except Exception as e:
        print(f"❌ Error getting daily summaries: {e}")
        return []

# ========================================
# SYSTEM LOG FUNCTIONS
# ========================================

def save_log(log_level, component, message, details=None):
    """Save system log"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        details_json = json.dumps(details) if details else None
        
        cursor.execute('''
            INSERT INTO system_logs (log_level, component, message, details)
            VALUES (?, ?, ?, ?)
        ''', (log_level, component, message, details_json))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error saving log: {e}")
        return False

# ========================================
# UTILITY FUNCTIONS
# ========================================

def get_database_stats():
    """Get overall database statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count records in each table
        cursor.execute("SELECT COUNT(*) FROM sensor_readings")
        stats['sensor_readings'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM detections")
        stats['detections'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM relay_events")
        stats['relay_events'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM daily_summary")
        stats['daily_summary'] = cursor.fetchone()[0]
        
        # Get date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM sensor_readings")
        row = cursor.fetchone()
        stats['first_reading'] = row[0]
        stats['last_reading'] = row[1]
        
        # Get database file size
        stats['db_size_mb'] = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0
        
        conn.close()
        return stats
    except Exception as e:
        print(f"❌ Error getting database stats: {e}")
        return {}

def export_to_csv(table_name, output_file):
    """Export table to CSV file"""
    try:
        import csv
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT * FROM {table_name}")
        
        rows = cursor.fetchall()
        headers = [description[0] for description in cursor.description]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        conn.close()
        print(f"✅ Exported {len(rows)} rows from {table_name} to {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error exporting to CSV: {e}")
        return False

# Initialize database when module is imported
if __name__ == "__main__":
    init_database()
else:
    # Auto-initialize if database doesn't exist
    if not os.path.exists(DB_PATH):
        init_database()
