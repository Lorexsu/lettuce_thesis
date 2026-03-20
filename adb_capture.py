import subprocess
import numpy as np
import cv2
import time
import os

class ADBCapture:
    def __init__(self, device_id=None):
        """
        Initialize ADB capture
        device_id: Optional device ID (e.g., "emulator-5554" or "127.0.0.1:7555")
                   If None, uses first device found
        """
        # Set the full path to adb.exe (update this if your path is different)
        self.adb_path = r"C:\Users\Rolex Jr\Downloads\platform-tools\adb.exe"
        
        # Check if adb exists
        if not os.path.exists(self.adb_path):
            # Try without full path (if in PATH)
            self.adb_path = "adb"
            
        self.device_id = device_id
        self.device = self._get_device()
        print(f"[ADBCapture] Using device: {self.device}")
        print(f"[ADBCapture] Using adb: {self.adb_path}")
        
    def _get_device(self):
        """Get the target device ID"""
        if self.device_id:
            return self.device_id
        
        # Get list of devices
        try:
            result = subprocess.run([self.adb_path, 'devices'], 
                                    capture_output=True, text=True, check=True)
        except:
            # Try with just "adb" if path didn't work
            result = subprocess.run(['adb', 'devices'], 
                                    capture_output=True, text=True, check=True)
            
        lines = result.stdout.strip().split('\n')[1:]  # Skip first line
        
        devices = []
        for line in lines:
            if '\tdevice' in line:
                device = line.split('\t')[0]
                devices.append(device)
        
        if not devices:
            raise Exception("No Android device/emulator found")
        
        # If multiple devices, prefer emulator-5554 (typical emulator)
        for device in devices:
            if 'emulator-5554' in device:
                return device
        
        # Otherwise return first device
        return devices[0]
    
    def capture_frame(self):
        """Capture a single frame via ADB screencap"""
        try:
            # Try with full adb path first
            try:
                cmd = [self.adb_path, '-s', self.device, 'exec-out', 'screencap', '-p']
                result = subprocess.run(cmd, capture_output=True, check=True)
            except:
                # Fallback to just "adb"
                cmd = ['adb', '-s', self.device, 'exec-out', 'screencap', '-p']
                result = subprocess.run(cmd, capture_output=True, check=True)
            
            if result.returncode != 0:
                print(f"[ADBCapture] Error: {result.stderr}")
                return None
            
            # Convert PNG bytes to numpy array (OpenCV format)
            img_bytes = result.stdout
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Resize for better performance (optional)
            if frame is not None and frame.shape[1] > 800:
                scale = 800 / frame.shape[1]
                new_width = 800
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            return frame
            
        except subprocess.CalledProcessError as e:
            print(f"[ADBCapture] ADB error: {e}")
            return None
        except Exception as e:
            print(f"[ADBCapture] Capture error: {e}")
            return None
    
    def get_frame(self):
        """Alias for capture_frame (compatibility)"""
        return self.capture_frame()


# For testing
if __name__ == "__main__":
    print("Testing ADB Capture...")
    print("Make sure your emulator is running!")
    
    try:
        cap = ADBCapture()
        print(f"\n✅ Connected to {cap.device}")
        print("\nCapturing frames (press 'q' to quit)...")
        print("(This may be slow at first - wait for frames to appear)\n")
        
        frame_count = 0
        start_time = time.time()
        last_fps_time = start_time
        
        while True:
            frame = cap.capture_frame()  # Using capture_frame method
            if frame is not None:
                frame_count += 1
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Calculate FPS every second
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"✅ Frames: {frame_count} | FPS: {fps:.1f}")
                    last_fps_time = current_time
                
                # Add info to frame
                cv2.putText(frame, f"Device: {cap.device}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('ADB Capture - Press Q to quit', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("⏳ Waiting for frame...")
                time.sleep(0.5)
        
        cv2.destroyAllWindows()
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n✅ Test complete. Captured {frame_count} frames in {total_time:.1f} seconds")
        print(f"   Average FPS: {avg_fps:.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure emulator is running")
        print(f"2. Check if adb is at: {cap.adb_path if 'cap' in locals() else 'unknown'}")
        print("3. Run 'adb devices' manually to verify connection")