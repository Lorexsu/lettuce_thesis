from flask import Flask, request, jsonify, send_from_directory, render_template, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
from datetime import datetime
import json
import cv2
import numpy as np
import threading
import requests
import time
import torch
import torch.nn as nn
from torchvision import transforms
import timm

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')


# Tapo Camera Configuration
TAPO_IP = "192.168.1.49"  
TAPO_USER = "topac200c"
TAPO_PASS = "c200ctopa"  
STREAM_URL = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream1"

# ESP32 Configuration
ESP32_IP = "10.0.0.42"
ESP32_SENSOR_URL = f"http://{ESP32_IP}:5000/get-sensor-data"    

#Load YOLO (Readiness Detection)
model_path = "best.pt"

try:
    model = YOLO(model_path)
    print(f"✅ Model 1 (YOLO) loaded from {model_path}")
except Exception as e:
    print(f"❌ Error loading Model 1: {e}")
    model = YOLO("yolo12s.pt")


# MODEL 2: Load Nutrient Stress Classifier
class NutrientStressClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

try:
    nutrient_model = NutrientStressClassifier(num_classes=4)
    nutrient_model.load_state_dict(torch.load('nutrient_stress_model.pth', map_location=torch.device('cpu')))
    nutrient_model.eval()
    print("✅ Model 2 (Nutrient Stress) loaded successfully")
    
    with open('model_metadata.json', 'r') as f:
        nutrient_metadata = json.load(f)
    print(f"   Classes: {nutrient_metadata['class_names']}")
    
    nutrient_model_loaded = True
except Exception as e:
    print(f"⚠️ Model 2 (Nutrient Stress) not loaded: {e}")
    nutrient_model_loaded = False

nutrient_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])


# GLOBAL STATE
sensor_data = {
    'temperature': None,
    'humidity': None,
    'timestamp': None
}
activity_logs = []
live_stream_active = False
latest_detection = None

# LIVE STREAM PROCESSING
def process_frame_yolo(frame):
    """Process single frame with YOLO"""
    results = model.predict(frame, conf=0.3, iou=0.3, verbose=False)
    
    detections = []
    annotated_frame = frame.copy()
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        annotated_frame = results[0].plot()
        
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = results[0].names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Size-based classification override
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            area_ratio = bbox_area / frame_area
            
            # If detection is small (<5% of frame), likely NOT ready
            if area_ratio < 0.02 and 'ready' in label.lower() and 'not' not in label.lower():
                label = 'Not Ready to Harvest (small size)'
                conf = conf * 0.8  # Lower confidence
            
            # Run health classification on detected lettuce
            health_status = 'Unknown'
            health_confidence = 0.0
            
            if nutrient_model_loaded:
                try:
                    # Crop detected lettuce from frame
                    y1_int = max(0, int(y1))
                    y2_int = min(frame.shape[0], int(y2))
                    x1_int = max(0, int(x1))
                    x2_int = min(frame.shape[1], int(x2))
                    
                    lettuce_crop = frame[y1_int:y2_int, x1_int:x2_int]
                    
                    if lettuce_crop.size > 0 and lettuce_crop.shape[0] > 10 and lettuce_crop.shape[1] > 10:
                        # Convert BGR to RGB
                        rgb_crop = cv2.cvtColor(lettuce_crop, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(rgb_crop)
                        
                        # Transform and predict
                        input_tensor = nutrient_transform(pil_crop).unsqueeze(0)
                        
                        with torch.no_grad():
                            output = nutrient_model(input_tensor)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            health_conf, health_pred = torch.max(probabilities, 1)
                        
                        health_status = nutrient_metadata['class_names'][health_pred.item()]
                        health_confidence = float(health_conf.item())
                except Exception as e:
                    print(f"Health classification error: {e}")
                    health_status = 'Error'
            
            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'health_status': health_status,
                'health_confidence': health_confidence
            })
    
    return annotated_frame, detections

def live_stream_processor():
    """Background thread to process ESP32 stream"""
    global live_stream_active, latest_detection
    
    print(f"🎥 Starting live stream from {STREAM_URL}")
    
    cap = cv2.VideoCapture(STREAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"❌ FAILED to open camera stream at {STREAM_URL}")
        print("   Check: 1) Camera IP reachable 2) RTSP port 554 open 3) Credentials correct")
        socketio.emit('stream_status', {'status': 'error', 'message': 'Camera connection failed'})
        return
    else:
        print(f"✅ Camera stream opened successfully")
    
    retry_count = 0
    max_retries = 5
    
    while live_stream_active:
        try:
            if not cap.isOpened():
                print(f"Reconnecting to stream... (attempt {retry_count + 1})")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(STREAM_URL)
                retry_count += 1
                if retry_count >= max_retries:
                    print("Max retries reached. Stopping stream.")
                    break
                continue
            
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print("Failed to read frame. Retrying...")
                time.sleep(0.5)
                continue
            
            retry_count = 0
            annotated_frame, detections = process_frame_yolo(frame)
            
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
                        
            # Prepare detection data
            if detections:
                ready_count = sum(1 for d in detections if 'ready' in d['label'].lower() and 'not' not in d['label'].lower())
                not_ready_count = len(detections) - ready_count
                
                result = {
                    "frame": f"data:image/jpeg;base64,{frame_base64}",
                    "total_count": len(detections),
                    "ready_count": ready_count,
                    "not_ready_count": not_ready_count,
                    "detections": detections,
                    "timestamp": time.time()
                }
            else:
                result = {
                    "frame": f"data:image/jpeg;base64,{frame_base64}",
                    "total_count": 0,
                    "ready_count": 0,
                    "not_ready_count": 0,
                    "detections": [],
                    "timestamp": time.time()
                }
            
            latest_detection = result
            socketio.emit('detection_update', result)
            time.sleep(0.033)  # ~30 FPS
                        
        except Exception as e:
            print(f"Stream error: {e}")
            cap.release()
            time.sleep(2)
    
    cap.release()


# WEBSOCKET EVENTS
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    if latest_detection:
        emit('detection_update', latest_detection)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_live_stream')
def handle_start_stream():
    print('🔵 Received start_live_stream event')
    global live_stream_active
    if not live_stream_active:
        print('🔵 Starting new stream thread...')
        live_stream_active = True
        thread = threading.Thread(target=live_stream_processor, daemon=True)
        thread.start()
        emit('stream_status', {'status': 'started'})
        print('✅ Live stream started')
    else:
        print('⚠️ Stream already active')

@socketio.on('stop_live_stream')
def handle_stop_stream():
    global live_stream_active
    live_stream_active = False
    emit('stream_status', {'status': 'stopped'})
    print('⏹️ Live stream stopped')


# ROUTES (Keep existing + add live view)
#@app.route('/')
#def index():
    #return send_from_directory('templates', 'lettuce_home_new.html')

@app.route('/')
def index():
    return send_from_directory('templates', 'lettuce_dashboard.html')

@app.route('/history')
def history_page():
    return send_from_directory('templates', 'lettuce_history_lettuce.html')

@app.route('/test')
def test_page():
    return send_from_directory('templates', 'lettuce_test_lettuce.htm')

@app.route('/about')
def about_page():
    return send_from_directory('templates', 'lettuce_about_lettuce.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('templates', filename)

@app.route('/styles.css')
def serve_styles():
    return send_from_directory('static', 'styles.css')


# EXISTING ENDPOINTS (unchanged)
@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        results = model.predict(image, conf=0.4)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            annotated_image = results[0].plot()
            
            from PIL import Image as PILImage
            annotated_pil = PILImage.fromarray(annotated_image)
            buffered = io.BytesIO()
            annotated_pil.save(buffered, format="JPEG")
            annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            all_detections = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = results[0].names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                all_detections.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                })
            
            best_box = max(boxes, key=lambda b: b.conf[0])
            cls_id = int(best_box.cls[0].item())
            conf = float(best_box.conf[0].item())
            label = results[0].names[cls_id]
            
            total_count = len(boxes)
            avg_confidence = sum(d['confidence'] for d in all_detections) / total_count
            ready_count = sum(1 for d in all_detections if 'ready' in d['label'].lower() and 'not' not in d['label'].lower())
            not_ready_count = sum(1 for d in all_detections if 'not' in d['label'].lower() or 'seedling' in d['label'].lower())
            
            return jsonify({
                'detected': True,
                'classification': label,
                'confidence': conf,
                'count': total_count,
                'average_confidence': avg_confidence,
                'ready_count': ready_count,
                'not_ready_count': not_ready_count,
                'all_detections': all_detections,
                'annotated_image': f'data:image/jpeg;base64,{annotated_base64}'
            })
        else:
            return jsonify({'detected': False})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'detected': False, 'error': str(e)}), 500

@app.route('/classify-nutrient', methods=['POST'])
def classify_nutrient():
    if not nutrient_model_loaded:
        return jsonify({
            'detected': False, 
            'error': 'Nutrient stress model not loaded'
        }), 503
    
    try:
        data = request.json
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        input_tensor = nutrient_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = nutrient_model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        nutrient_class = nutrient_metadata['class_names'][predicted.item()]
        
        all_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(
                nutrient_metadata['class_names'], 
                probabilities[0].tolist()
            )
        }
        
        return jsonify({
            'detected': True,
            'nutrient_status': nutrient_class,
            'confidence': float(confidence.item()),
            'all_probabilities': all_probabilities
        })
    except Exception as e:
        print(f"Nutrient detection error: {e}")
        return jsonify({'detected': False, 'error': str(e)}), 500

#Sensor Data Storage

sensor_history = {
    'temperature': [],
    'humidity': [],
    'timestamps': []
}
MAX_HISTORY_SIZE = 1000  # Store last 1000 readings

@app.route('/sensor-data', methods=['POST'])
def receive_sensor_data():
    try:
        data = request.json
        
        # Update current sensor data
        sensor_data['temperature'] = data.get('temperature')
        sensor_data['humidity'] = data.get('humidity')
        sensor_data['timestamp'] = data.get('timestamp')
        sensor_data['sensor_id'] = data.get('sensor_id', 'unknown')
        sensor_data['location'] = data.get('location', 'unknown')
        
        # ✅ FIX: Validate timestamp before adding to history
        timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Check if timestamp is from today and not stuck in the past
        try:
            ts_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            now = datetime.now()
            
            # If timestamp is more than 1 hour old OR from a different day, use current time instead
            if (now - ts_date).total_seconds() > 3600 or ts_date.date() != now.date():
                print(f"⚠️ Old timestamp detected: {timestamp}, replacing with current time")
                timestamp = now.isoformat()
        except:
            # If timestamp parsing fails, use current time
            timestamp = datetime.now().isoformat()
        
        # Add to history with validated timestamp
        if data.get('temperature') is not None:
            sensor_history['temperature'].append(data['temperature'])
            sensor_history['humidity'].append(data['humidity'])
            sensor_history['timestamps'].append(timestamp)
        
        # Keep history size manageable
        if len(sensor_history['temperature']) > MAX_HISTORY_SIZE:
            sensor_history['temperature'] = sensor_history['temperature'][-MAX_HISTORY_SIZE:]
            sensor_history['humidity'] = sensor_history['humidity'][-MAX_HISTORY_SIZE:]
            sensor_history['timestamps'] = sensor_history['timestamps'][-MAX_HISTORY_SIZE:]
        
        # Log the reception
        print(f"📊 Sensor Data Received: {data['temperature']}°C, {data['humidity']}% at {timestamp}")
        
        # Run automation checks
        check_automation()
        
        return jsonify({'status': 'success', 'message': 'Data received'})
    except Exception as e:
        print(f"Error receiving sensor data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get-sensor-data', methods=['GET'])
def get_sensor_data():
    """Get current sensor reading"""
    try:
        return jsonify(sensor_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-sensor-history', methods=['GET'])
def get_sensor_history():
    """Get historical sensor data for charts"""
    try:
        limit = request.args.get('limit', 100, type=int)
        
        # Return last N readings
        history = {
            'temperature': sensor_history['temperature'][-limit:],
            'humidity': sensor_history['humidity'][-limit:],
            'timestamps': sensor_history['timestamps'][-limit:]
        }
        
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sensor-stats', methods=['GET'])
def get_sensor_stats():
    """Get sensor statistics"""
    try:
        if len(sensor_history['temperature']) == 0:
            return jsonify({
                'avg_temp': None,
                'avg_humid': None,
                'min_temp': None,
                'max_temp': None,
                'min_humid': None,
                'max_humid': None
            })
        
        temps = sensor_history['temperature'][-24:]  
        humids = sensor_history['humidity'][-24:]    
        
        stats = {
            'avg_temp': round(sum(temps) / len(temps), 1),
            'avg_humid': round(sum(humids) / len(humids), 1),
            'min_temp': round(min(temps), 1),
            'max_temp': round(max(temps), 1),
            'min_humid': round(min(humids), 1),
            'max_humid': round(max(humids), 1),
            'reading_count': len(sensor_history['temperature']),
            'latest_reading': sensor_data['timestamp']
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Relay states
relay_states = {
    'fan': False,
    'light': False,
    'pump': False
}

# Automation rules
TEMP_THRESHOLD = 30.0
HUMIDITY_THRESHOLD = 65.0
PUMP_DURATION = 4  # seconds
PUMP_COOLDOWN = 3600  # seconds (1 hour)
LIGHT_START_HOUR = 18  # 6 PM
LIGHT_START_MINUTE = 0
LIGHT_END_HOUR = 6     # 6 AM next day

# Pump timing state
pump_state = {
    'last_activation': None,
    'activation_end': None
}

def check_automation():
    """Check sensor data and update relays based on rules"""
    global relay_states
    
    temp = sensor_data.get('temperature')
    humid = sensor_data.get('humidity')
    
    changes_made = False
    
    # Fan automation (ON when temp >= 25°C)
    if temp is not None:
        should_activate_fan = temp >= TEMP_THRESHOLD
        if relay_states['fan'] != should_activate_fan:
            relay_states['fan'] = should_activate_fan
            log_activity('Fan', 'ON' if should_activate_fan else 'OFF', 
                        f'Temperature {temp}°C', temp, humid)
            print(f"🌀 Fan {'ON' if should_activate_fan else 'OFF'} - Temp: {temp}°C")
            changes_made = True
    
    # Pump automation with timed activation and cooldown
    if humid is not None:
        from datetime import datetime, timedelta
        now = datetime.now()
        
        # Check if pump is currently running (within 4 second window)
        if pump_state['activation_end'] and now < pump_state['activation_end']:
            relay_states['pump'] = True
            print(f"💧 Pump RUNNING - {(pump_state['activation_end'] - now).seconds}s remaining")
        
        # Check if in cooldown period
        elif pump_state['last_activation'] and (now - pump_state['last_activation']).total_seconds() < PUMP_COOLDOWN:
            relay_states['pump'] = False
            cooldown_remaining = PUMP_COOLDOWN - (now - pump_state['last_activation']).total_seconds()
            print(f"💧 Pump COOLDOWN - {int(cooldown_remaining)}s remaining")
        
        # Check if should activate (threshold reached and not in cooldown)
        elif humid > HUMIDITY_THRESHOLD:
            if not relay_states['pump']:  # Only log on new activation
                relay_states['pump'] = True
                pump_state['last_activation'] = now
                pump_state['activation_end'] = now + timedelta(seconds=PUMP_DURATION)
                log_activity('Pump', 'ON', f'Humidity {humid}% - Running for {PUMP_DURATION}s', temp, humid)
                print(f"💧 Pump ACTIVATED - Humidity: {humid}% - Running for {PUMP_DURATION}s")
        
        else:
            # Below threshold and not running
            relay_states['pump'] = False
    
    # Light automation (ON from 5:50 PM to 6:00 AM)
    from datetime import datetime
    now = datetime.now()
    current_time = now.hour * 60 + now.minute
    start_time = LIGHT_START_HOUR * 60 + LIGHT_START_MINUTE
    end_time = LIGHT_END_HOUR * 60
    
    if start_time > end_time:
        should_activate_light = current_time >= start_time or current_time < end_time
    else:
        should_activate_light = start_time <= current_time < end_time
    
    if relay_states['light'] != should_activate_light:
        relay_states['light'] = should_activate_light
        log_activity('Light', 'ON' if should_activate_light else 'OFF',
                    f'Scheduled time {now.strftime("%H:%M")}', temp, humid)
        print(f"💡 Light {'ON' if should_activate_light else 'OFF'} - Time: {now.strftime('%H:%M')}")
        changes_made = True
    
    if changes_made:
        print(f"✅ Relay states updated - Fan: {relay_states['fan']}, Light: {relay_states['light']}, Pump: {relay_states['pump']}")

def log_activity(device, action, reason, temp=None, humid=None):
    """Log relay activity"""
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'action': action,
        'reason': reason,
        'temperature': temp,
        'humidity': humid,
        'battery_voltage': 'N/A'
    }
    activity_logs.append(log_entry)
    if len(activity_logs) > 1000:
        activity_logs.pop(0)

@app.route('/get-relay-states', methods=['GET'])
def get_relay_states():
    """Get all relay states for ESP32"""
    return jsonify(relay_states)

@app.route('/set-relay', methods=['POST'])
def set_relay():
    """Manual relay control from dashboard"""
    try:
        data = request.json
        device = data.get('device')  # 'fan', 'light', or 'pump'
        state = data.get('state', False)
        
        if device in relay_states:
            relay_states[device] = state
            temp = sensor_data.get('temperature')
            humid = sensor_data.get('humidity')
            log_activity(device.capitalize(), 'ON' if state else 'OFF', 
                        'Manual override', temp, humid)
            print(f"🔌 {device.upper()} {'ON' if state else 'OFF'} (Manual)")
            return jsonify({'status': 'success', 'device': device, 'state': state})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid device'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
@app.route('/log-activity', methods=['POST'])
def log_activity():
    try:
        data = request.json
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': data.get('device'),
            'action': data.get('action'),
            'reason': data.get('reason'),
            'temperature': data.get('temperature'),
            'humidity': data.get('humidity'),
            'battery_voltage': data.get('battery_voltage', 'N/A')
        }
        activity_logs.append(log_entry)
        if len(activity_logs) > 1000:
            activity_logs.pop(0)
        return jsonify({'status': 'success', 'log_id': len(activity_logs)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get-activity-logs', methods=['GET'])
def get_activity_logs():
    limit = request.args.get('limit', 100, type=int)
    return jsonify({'logs': activity_logs[-limit:]})

@app.route('/clear-logs', methods=['POST'])
def clear_logs():
    global activity_logs
    activity_logs = []
    return jsonify({'status': 'success', 'message': 'All logs cleared'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model1_loaded': model is not None,
        'model2_loaded': nutrient_model_loaded,
        'total_logs': len(activity_logs),
        'live_stream_active': live_stream_active
    })

@app.route('/get-detection-history', methods=['GET'])
def get_detection_history():
    """Get historical detection data for charts"""
    try:
        # In production, you'd fetch from a database
        # For now, return sample data or empty
        return jsonify({
            'growth': {
                'labels': [],
                'ready': [],
                'notReady': []
            },
            'daily': {
                'labels': [],
                'ready': [],
                'notReady': []
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-detection-data', methods=['POST'])
def save_detection_data():
    """Save detection data with temperature and humidity"""
    try:
        data = request.json
        
        # Add current sensor readings to analytics data
        if 'analytics' in data:
            temp = sensor_data.get('temperature')
            humid = sensor_data.get('humidity')
            
            if temp is not None and humid is not None:
                env_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'temperature': temp,
                    'humidity': humid
                }
                
                if 'environmentData' not in data['analytics']:
                    data['analytics']['environmentData'] = []
                
                data['analytics']['environmentData'].append(env_entry)
        
        # Save to file with timestamp
        filename = f"lettuce-analytics-{datetime.now().strftime('%Y-%m-%d')}.json"
        filepath = os.path.join('analytics_data', filename)
        
        os.makedirs('analytics_data', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Analytics saved to {filename}")
        
        return jsonify({'status': 'success', 'message': 'Data saved', 'file': filename})
    except Exception as e:
        print(f"Error saving analytics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get-pump-settings', methods=['GET'])
def get_pump_settings():
    """Get current pump settings"""
    return jsonify({
        'duration': PUMP_DURATION,
        'cooldown': PUMP_COOLDOWN,
        'threshold': HUMIDITY_THRESHOLD
    })

@app.route('/set-pump-settings', methods=['POST'])
def set_pump_settings():
    """Update pump settings"""
    global PUMP_DURATION, PUMP_COOLDOWN, HUMIDITY_THRESHOLD
    try:
        data = request.json
        if 'duration' in data:
            PUMP_DURATION = int(data['duration'])
        if 'cooldown' in data:
            PUMP_COOLDOWN = int(data['cooldown'])
        if 'threshold' in data:
            HUMIDITY_THRESHOLD = float(data['threshold'])
        
        print(f"⚙️ Pump settings updated: Duration={PUMP_DURATION}s, Cooldown={PUMP_COOLDOWN}s, Threshold={HUMIDITY_THRESHOLD}%")
        return jsonify({
            'status': 'success',
            'duration': PUMP_DURATION,
            'cooldown': PUMP_COOLDOWN,
            'threshold': HUMIDITY_THRESHOLD
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🥬 Lettuce Monitoring System - LIVE STREAM Edition")
    print("="*50)
    print("✅ Model 1 (YOLO): Readiness Detection")
    print(f"✅ Model 2 (Nutrient): {'Loaded' if nutrient_model_loaded else 'Not Available'}")
    print(f"📹 ESP32 Stream: {STREAM_URL}")
    print("✅ Open: http://localhost:5000")
    print("="*50 + "\n")

    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
