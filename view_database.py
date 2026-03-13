"""
Simple script to view your SQLite database contents
Run this to see what's in your database!
"""

import sqlite3
from datetime import datetime
import os

DB_PATH = 'C:/Users/Rolex Jr/OneDrive/Desktop/lettuce_model_thesis/ui-for-lettuce-thesis-1.5/lettuce_farm.db'

def print_separator():
    print("\n" + "="*80 + "\n")

def view_database():
    """Display database contents in a readable format"""
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        print("Run 'python database.py' first to create the database!")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("\n" + "🗄️  LETTUCE FARM DATABASE VIEWER ".center(80, "="))
    print(f"📁 Database: {os.path.abspath(DB_PATH)}")
    print(f"💾 Size: {os.path.getsize(DB_PATH) / 1024:.2f} KB")
    
    print_separator()
    
    # ========================================
    # DATABASE STATISTICS
    # ========================================
    print("📊 DATABASE STATISTICS".center(80))
    print_separator()
    
    tables = ['sensor_readings', 'detections', 'relay_events', 'daily_summary', 'system_logs']
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table.ljust(20)} : {count:,} records")
    
    print_separator()
    
    # ========================================
    # LATEST SENSOR READINGS (Last 10)
    # ========================================
    print("🌡️  LATEST SENSOR READINGS (Last 10)".center(80))
    print_separator()
    
    cursor.execute('''
        SELECT timestamp, temperature, humidity, sensor_id, location
        FROM sensor_readings
        ORDER BY timestamp DESC
        LIMIT 10
    ''')
    
    rows = cursor.fetchall()
    
    if rows:
        print(f"{'Timestamp':<20} {'Temp (°C)':<12} {'Humidity (%)':<15} {'Sensor ID':<15} {'Location'}")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]:<20} {row[1]:<12.1f} {row[2]:<15.1f} {row[3]:<15} {row[4]}")
    else:
        print("  No sensor readings yet")
    
    print_separator()
    
    # ========================================
    # DETECTION SUMMARY
    # ========================================
    print("🥬 DETECTION SUMMARY (Last 7 Days)".center(80))
    print_separator()
    
    cursor.execute('''
        SELECT 
            DATE(timestamp) as date,
            label,
            COUNT(*) as count,
            AVG(confidence) as avg_conf
        FROM detections
        WHERE timestamp >= datetime('now', '-7 days')
        GROUP BY DATE(timestamp), label
        ORDER BY date DESC, label
    ''')
    
    rows = cursor.fetchall()
    
    if rows:
        print(f"{'Date':<15} {'Label':<35} {'Count':<10} {'Avg Conf'}")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]:<15} {row[1]:<35} {row[2]:<10} {row[3]:.2%}")
    else:
        print("  No detections yet")
    
    print_separator()
    
    # ========================================
    # RELAY EVENTS (Last 20)
    # ========================================
    print("⚡ RELAY EVENTS (Last 20)".center(80))
    print_separator()
    
    cursor.execute('''
        SELECT timestamp, relay_name, action, trigger_type, temperature, humidity
        FROM relay_events
        ORDER BY timestamp DESC
        LIMIT 20
    ''')
    
    rows = cursor.fetchall()
    
    if rows:
        print(f"{'Timestamp':<20} {'Relay':<10} {'Action':<8} {'Trigger':<15} {'Temp':<8} {'Humid'}")
        print("-" * 80)
        for row in rows:
            temp_str = f"{row[4]:.1f}°C" if row[4] else "N/A"
            humid_str = f"{row[5]:.1f}%" if row[5] else "N/A"
            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<8} {row[3]:<15} {temp_str:<8} {humid_str}")
    else:
        print("  No relay events yet")
    
    print_separator()
    
    # ========================================
    # DAILY SUMMARIES
    # ========================================
    print("📅 DAILY SUMMARIES".center(80))
    print_separator()
    
    cursor.execute('''
        SELECT date, total_detections, ready_count, not_ready_count,
               avg_temperature, avg_humidity
        FROM daily_summary
        ORDER BY date DESC
        LIMIT 10
    ''')
    
    rows = cursor.fetchall()
    
    if rows:
        print(f"{'Date':<15} {'Total':<10} {'Ready':<10} {'Not Ready':<12} {'Avg Temp':<12} {'Avg Humid'}")
        print("-" * 80)
        for row in rows:
            temp_str = f"{row[4]:.1f}°C" if row[4] else "N/A"
            humid_str = f"{row[5]:.1f}%" if row[5] else "N/A"
            print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<12} {temp_str:<12} {humid_str}")
    else:
        print("  No daily summaries yet")
    
    print_separator()
    
    # ========================================
    # QUICK STATS
    # ========================================
    print("📈 QUICK STATS".center(80))
    print_separator()
    
    # Total detections
    cursor.execute("SELECT COUNT(*) FROM detections")
    total_detections = cursor.fetchone()[0]
    
    # Ready vs Not Ready
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN label LIKE '%Ready%' AND label NOT LIKE '%Not%' THEN 1 ELSE 0 END) as ready,
            SUM(CASE WHEN label LIKE '%Not Ready%' OR label LIKE '%small size%' THEN 1 ELSE 0 END) as not_ready
        FROM detections
    """)
    ready_counts = cursor.fetchone()
    
    # Average temperature and humidity
    cursor.execute("""
        SELECT AVG(temperature), AVG(humidity)
        FROM sensor_readings
        WHERE timestamp >= datetime('now', '-24 hours')
    """)
    avg_row = cursor.fetchone()
    
    print(f"  Total Detections (all time)    : {total_detections:,}")
    print(f"  Ready to Harvest (all time)    : {ready_counts[0] or 0:,}")
    print(f"  Not Ready (all time)           : {ready_counts[1] or 0:,}")
    
    if avg_row[0]:
        print(f"  Avg Temperature (24h)          : {avg_row[0]:.1f}°C")
    if avg_row[1]:
        print(f"  Avg Humidity (24h)             : {avg_row[1]:.1f}%")
    
    # Relay usage
    cursor.execute("""
        SELECT relay_name, COUNT(*) as count
        FROM relay_events
        WHERE action = 'ON'
        GROUP BY relay_name
        ORDER BY count DESC
    """)
    
    relay_counts = cursor.fetchall()
    if relay_counts:
        print(f"\n  Relay Activations (all time):")
        for relay, count in relay_counts:
            print(f"    {relay.capitalize():<15} : {count:,} times")
    
    print_separator()
    
    conn.close()
    
    print("\n✅ Database viewed successfully!")
    print("\n💡 Tips:")
    print("  - Use DB Browser for SQLite to view visually")
    print("  - Download: https://sqlitebrowser.org/")
    print("  - Or use sqlite3 command line tools")
    print("\n")

def export_all_data():
    """Export all tables to CSV files"""
    from database import export_to_csv
    
    tables = ['sensor_readings', 'detections', 'relay_events', 'daily_summary']
    
    print("\n📤 EXPORTING DATA TO CSV FILES")
    print_separator()
    
    for table in tables:
        filename = f"{table}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        if export_to_csv(table, filename):
            print(f"✅ Exported: {filename}")
    
    print_separator()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'export':
        export_all_data()
    else:
        view_database()
