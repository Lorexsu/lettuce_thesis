import os
import csv
import glob
import gzip
import sqlite3
 
from database import DB_PATH, init_database, update_daily_summary
 
SEED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'seed_data')
 
 
def _num(v):
    return float(v) if v not in (None, '') else None
 
 
def _seed_table(conn, table, pattern, sql, row_fn, batch=20000):
    """Import newest seed_data/<pattern> (.csv or .csv.gz) only if the table is empty."""
    if conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0:
        print(f"⏭️  {table}: already has data - skipped")
        return 0
    files = sorted(glob.glob(os.path.join(SEED_DIR, pattern)))
    if not files:
        print(f"⚠️  {table}: no file matching seed_data/{pattern}")
        return 0
    path = files[-1]
    opener = gzip.open if path.endswith('.gz') else open
    total, rows, skipped = 0, [], 0
    with opener(path, 'rt', newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                rows.append(row_fn(r))
            except (ValueError, KeyError):
                skipped += 1
                continue
            if len(rows) >= batch:
                conn.executemany(sql, rows)
                total += len(rows)
                rows = []
    if rows:
        conn.executemany(sql, rows)
        total += len(rows)
    conn.commit()
    print(f"🌱 Seeded {total:,} rows into {table} from {os.path.basename(path)}"
          + (f" ({skipped} bad rows skipped)" if skipped else ""))
    return total
 
 
def _sensor_row(r):
    return (int(r['id']), r['timestamp'], _num(r['temperature']), _num(r['humidity']),
            r['sensor_id'], r['location'])
 
 
def _relay_row(r):
    return (int(r['id']), r['timestamp'], r['relay'], r['action'], r['trigger'],
            _num(r['temperature']), _num(r['humidity']))
 
 
def _detection_row(r):
    x1, y1, x2, y2 = [float(v) for v in r['bbox'].strip('() ').split(',')]
    return (int(r['id']), r['timestamp'], r['label'], _num(r['confidence']),
            x1, y1, x2, y2, r['health_status'], _num(r['health_confidence']), '')
 
 
def main():
    if not os.path.isdir(SEED_DIR):
        print(f"❌ seed_data folder not found: {SEED_DIR}")
        return
 
    init_database()  # make sure the tables exist
    print(f"📁 Seeding database: {os.path.abspath(DB_PATH)}")
 
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        _seed_table(conn, 'sensor_readings', 'sensors*.csv*',
            "INSERT OR IGNORE INTO sensor_readings (id, timestamp, temperature, humidity, sensor_id, location) VALUES (?,?,?,?,?,?)",
            _sensor_row)
        _seed_table(conn, 'relay_events', 'relays*.csv*',
            "INSERT OR IGNORE INTO relay_events (id, timestamp, relay_name, action, trigger_type, temperature, humidity) VALUES (?,?,?,?,?,?,?)",
            _relay_row)
        _seed_table(conn, 'detections', 'detections*.csv*',
            "INSERT OR IGNORE INTO detections (id, timestamp, label, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, health_status, health_confidence, image_path) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            _detection_row)
 
        if conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0] == 0:
            dates = sorted(r[0] for r in conn.execute(
                "SELECT DATE(timestamp) FROM detections UNION SELECT DATE(timestamp) FROM sensor_readings"
            ) if r[0])
            for d in dates:
                update_daily_summary(d)
            if dates:
                print(f"🌱 Built daily_summary for {len(dates)} days")
    finally:
        conn.close()
 
    print("✅ Done")
 
 
if __name__ == "__main__":
    main()