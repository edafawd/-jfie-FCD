#!/usr/bin/env python3
"""
============================================================
Robot Car Brain - Raspberry Pi 5  (single file)
Wireless phone camera + YOLOv8n + floor-color obstacle avoidance
+ Ollama vision narration + Flask dashboard
============================================================
Reflex brain (fast):  CV floor analysis + YOLO  -> drives motors
Thinking brain (slow): Ollama vision model       -> narrates scene on dashboard

Camera: phone IP-camera stream (e.g. Android "IP Webcam" app)
Run:        python3 robot_brain.py
Options:    --no-ai  (disable Ollama)   --no-web   --port /dev/ttyUSB0
Dashboard:  http://<pi-ip>:5000
============================================================
"""

import cv2
import serial
import serial.tools.list_ports
import threading
import time
import argparse
import logging
import base64
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from ultralytics import YOLO
from flask import Flask, Response, render_template_string

# ============================================================
# Config
# ============================================================
SERIAL_BAUD      = 115200    # must match firmware Serial.begin()
INIT_SPEED       = 160       # firmware default is 220; calmer for indoor testing

# Map internal command names -> this firmware's short codes (F/B/L/R/S protocol)
CMD_MAP = {
    "FORWARD": "F", "SLOW_FORWARD": "F", "BACKWARD": "B",
    "LEFT": "L", "RIGHT": "R", "STOP": "S",
}

# Phone IP-camera. Give the BASE url; the camera thread auto-finds the stream path.
CAMERA_SOURCE    = "http://192.168.86.61:8080"
CAMERA_PATHS     = ["/video", "/videofeed", "/mjpegfeed", ""]   # tried in order
FRAME_W          = 640
FRAME_H          = 480
WEB_PORT         = 5000
DETECT_INTERVAL  = 0.08    # seconds between YOLO runs
DRIVE_INTERVAL   = 0.12    # seconds between Arduino commands
CONF_THRESH      = 0.40

# Stop the car if no fresh frame within this many seconds (wireless dropout)
FRAME_STALE_SEC  = 1.0

# Vision obstacle zones (fraction of frame height)
OBSTACLE_ZONE_TOP    = 0.55
FLOOR_SAMPLE_TOP     = 0.80
OBSTACLE_THRESH_STOP = 0.45    # 45% blocked -> stop/avoid
OBSTACLE_THRESH_SLOW = 0.25    # 25% blocked -> slow

# ---- Ollama (thinking brain) ----
OLLAMA_ENABLE    = True
OLLAMA_HOST      = "http://localhost:11434"   # or your PC: "http://192.168.86.50:11434"
OLLAMA_MODEL     = "moondream"                 # small vision model (or "llava")
OLLAMA_INTERVAL  = 5.0                          # seconds between AI looks
OLLAMA_PROMPT    = (
    "You are the navigator of a small roving robot car looking forward. "
    "Decide where it should explore next. Reply on ONE line in exactly this format:\n"
    "DIRECTION: <forward|left|right|stop> | <short reason>\n"
    "Use 'stop' only if something is right in front. Otherwise pick the most "
    "interesting open direction to explore.")

# ---- AI steering (LLM influences direction only when path is open) ----
AI_STEER_ENABLE   = True
AI_SUGGESTION_TTL = 8.0    # seconds an LLM suggestion stays valid
AI_TURN_BURST     = 0.40   # seconds to curve toward the suggested side
AI_TURN_COOLDOWN  = 1.20   # seconds of straight driving between curves

INTERESTING = {
    0:  "person", 56: "chair", 57: "couch", 59: "bed", 60: "dining table",
    62: "tv / monitor", 63: "laptop", 67: "cell phone",
    72: "refrigerator", 73: "book",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("RobotBrain")

# ============================================================
# Shared State
# ============================================================
@dataclass
class RobotState:
    current_cmd:    str   = "STOP"
    detections:     list  = field(default_factory=list)
    obstacle_left:  float = 0.0
    obstacle_center:float = 0.0
    obstacle_right: float = 0.0
    frame:          Optional[np.ndarray] = None
    annotated_frame:Optional[np.ndarray] = None
    running:        bool  = True
    avoid_until:    float = 0.0
    turn_count:     int   = 0
    ir_left:        int   = 0
    ir_right:       int   = 0
    ai_text:        str   = "(starting AI...)"
    ai_suggestion:  str   = "FORWARD"   # LLM-chosen explore direction
    ai_suggestion_t:float = 0.0
    ai_steer_until: float = 0.0
    ai_steer_cool:  float = 0.0
    ai_steer_dir:   str   = "FORWARD"
    last_frame_t:   float = 0.0
    cam_ok:         bool  = False

state = RobotState()
frame_lock = threading.Lock()


def frame_is_fresh():
    return state.cam_ok and (time.time() - state.last_frame_t) < FRAME_STALE_SEC

# ============================================================
# Arduino Serial
# ============================================================
def find_arduino():
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(k in desc for k in ["arduino", "ch340", "ch341"]):
            return p.device
    for dev in ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1", "/dev/ttyACM1"]:
        try:
            s = serial.Serial(dev, SERIAL_BAUD, timeout=0.3); s.close(); return dev
        except: pass
    return None

class ArduinoSerial:
    def __init__(self, port):
        self.ser = None
        try:
            self.ser = serial.Serial(port, SERIAL_BAUD, timeout=1)
            time.sleep(2)
            log.info(f"Arduino on {port}")
            self.send(f"SPEED:{INIT_SPEED}")   # set a calm default speed
        except Exception as e:
            log.error(f"Serial failed: {e}")
        threading.Thread(target=self._reader, daemon=True).start()

    def send(self, cmd):
        code = CMD_MAP.get(cmd.strip().upper(), cmd.strip())   # translate to short codes
        if self.ser and self.ser.is_open:
            try: self.ser.write((code + "\n").encode())
            except: pass

    def _reader(self):
        while state.running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.5); continue
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if "IR_L:" in line:
                    parts = dict(p.split(":") for p in line.split(",") if ":" in p)
                    state.ir_left  = int(parts.get("IR_L", 0))
                    state.ir_right = int(parts.get("IR_R", 0))
            except: pass

# ============================================================
# Camera Thread (wireless phone stream, auto-reconnect)
# ============================================================
class CameraThread(threading.Thread):
    def __init__(self): super().__init__(daemon=True)

    def _candidates(self):
        src = CAMERA_SOURCE.rstrip("/")
        if src.endswith((".jpg", "/video", "/videofeed", "/mjpegfeed")):
            return [CAMERA_SOURCE]                      # already a full stream url
        return [src + p for p in CAMERA_PATHS]

    def _find_stream(self):
        for u in self._candidates():
            cap = cv2.VideoCapture(u)
            ok = False
            if cap.isOpened():
                for _ in range(8):
                    r, _f = cap.read()
                    if r: ok = True; break
                    time.sleep(0.1)
            cap.release()
            if ok:
                return u
            log.warning(f"camera path gave no frames: {u}")
        return None

    def run(self):
        url = None
        while state.running:
            if url is None:
                url = self._find_stream()
                if url is None:
                    state.cam_ok = False
                    log.error("No working camera stream found, retrying in 3s...")
                    time.sleep(3); continue
                log.info(f"Camera stream: {url}")
            cap = cv2.VideoCapture(url)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # keep newest frame -> less latency
            except: pass
            if not cap.isOpened():
                state.cam_ok = False
                log.error("Camera unreachable, re-searching in 2s...")
                url = None; time.sleep(2); continue
            log.info("Camera connected.")
            fails = 0
            while state.running:
                ret, frame = cap.read()
                if not ret:
                    fails += 1
                    if fails > 30:                      # stream died -> reconnect
                        log.warning("Stream lost, reconnecting...")
                        break
                    time.sleep(0.03); continue
                fails = 0
                if frame.shape[1] != FRAME_W or frame.shape[0] != FRAME_H:
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                with frame_lock:
                    state.frame = frame.copy()
                    state.last_frame_t = time.time()
                    state.cam_ok = True
            state.cam_ok = False
            cap.release()
            time.sleep(1)

# ============================================================
# Vision Obstacle Analyzer (floor-color vs zone)
# ============================================================
class ObstacleAnalyzer:
    def __init__(self):
        self.floor_hsv_mean = None
        self.floor_hsv_std  = None
        self.calibrated     = False
        self.cal_frames     = 0

    def update(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.cal_frames < 20:
            floor_strip = hsv[int(h * FLOOR_SAMPLE_TOP):h, w//4: 3*w//4]
            if self.floor_hsv_mean is None:
                self.floor_hsv_mean = floor_strip.mean(axis=(0,1))
                self.floor_hsv_std  = floor_strip.std(axis=(0,1)) + 15
            else:
                a = 0.1
                self.floor_hsv_mean = (1-a)*self.floor_hsv_mean + a*floor_strip.mean(axis=(0,1))
                self.floor_hsv_std  = (1-a)*self.floor_hsv_std  + a*(floor_strip.std(axis=(0,1))+15)
            self.cal_frames += 1
            if self.cal_frames == 20:
                self.calibrated = True
                log.info(f"Floor calibrated: HSV mean={self.floor_hsv_mean.astype(int)}")

        if not self.calibrated:
            return 0.0, 0.0, 0.0

        y1 = int(h * OBSTACLE_ZONE_TOP)
        y2 = int(h * FLOOR_SAMPLE_TOP)
        zone = hsv[y1:y2, :]

        lo = np.clip(self.floor_hsv_mean - 2.5 * self.floor_hsv_std, 0, 255).astype(np.uint8)
        hi = np.clip(self.floor_hsv_mean + 2.5 * self.floor_hsv_std, 0, 255).astype(np.uint8)

        floor_mask    = cv2.inRange(zone, lo, hi)
        obstacle_mask = cv2.bitwise_not(floor_mask)

        third = w // 3
        left_fill   = obstacle_mask[:, :third].mean()        / 255.0
        center_fill = obstacle_mask[:, third:2*third].mean() / 255.0
        right_fill  = obstacle_mask[:, 2*third:].mean()      / 255.0

        state.obstacle_left   = float(left_fill)
        state.obstacle_center = float(center_fill)
        state.obstacle_right  = float(right_fill)
        return left_fill, center_fill, right_fill

    def draw_zones(self, frame):
        h, w = frame.shape[:2]
        y1 = int(h * OBSTACLE_ZONE_TOP)
        y2 = int(h * FLOOR_SAMPLE_TOP)
        third = w // 3

        def zone_color(fill):
            if fill > OBSTACLE_THRESH_STOP: return (0,0,255)
            if fill > OBSTACLE_THRESH_SLOW: return (0,165,255)
            return (0,200,0)

        overlay = frame.copy()
        for i, fill in enumerate([state.obstacle_left, state.obstacle_center, state.obstacle_right]):
            x1, x2 = i*third, (i+1)*third
            cv2.rectangle(overlay, (x1,y1),(x2,y2), zone_color(fill), -1)
            cv2.putText(frame, f"{fill:.0%}", (x1+5, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.rectangle(frame, (0,y1),(w,y2),(150,150,150),1)
        return frame

# ============================================================
# Detection + Vision Thread
# ============================================================
class VisionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        log.info("Loading YOLOv8n...")
        self.model    = YOLO("yolov8n.pt")
        self.analyzer = ObstacleAnalyzer()
        log.info("YOLOv8n ready.")

    def run(self):
        while state.running:
            with frame_lock:
                frame = state.frame
            if frame is None:
                time.sleep(0.04); continue

            self.analyzer.update(frame)

            results    = self.model(frame, conf=CONF_THRESH, verbose=False)[0]
            detections = []
            annotated  = self.analyzer.draw_zones(frame.copy())

            for box in results.boxes:
                cls_id = int(box.cls[0]); conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = INTERESTING.get(cls_id)
                if label is None:
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(80,80,80),1)
                    continue
                detections.append({
                    "label": label, "conf": conf, "box": (x1,y1,x2,y2),
                    "cx": (x1+x2)/2/FRAME_W, "cy": (y1+y2)/2/FRAME_H,
                    "close": y2 > FRAME_H * 0.55,
                })
                color = (0,255,80) if label == "person" else (0,200,255)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
                cv2.putText(annotated, f"{label} {conf:.0%}", (x1, max(y1-6,12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            state.detections = detections

            h, w = annotated.shape[:2]
            cv2.putText(annotated,
                f"CMD:{state.current_cmd}  L:{state.obstacle_left:.0%} C:{state.obstacle_center:.0%} R:{state.obstacle_right:.0%}",
                (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
            if not self.analyzer.calibrated:
                cv2.putText(annotated, "CALIBRATING FLOOR...", (w//2-100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

            with frame_lock:
                state.annotated_frame = annotated
            time.sleep(DETECT_INTERVAL)

# ============================================================
# Ollama Thread (thinking brain - narration only)
# ============================================================
class OllamaThread(threading.Thread):
    def __init__(self): super().__init__(daemon=True)
    def run(self):
        if not OLLAMA_ENABLE:
            state.ai_text = "(AI disabled)"; return
        while state.running:
            time.sleep(OLLAMA_INTERVAL)
            with frame_lock:
                frame = None if state.frame is None else state.frame.copy()
            if frame is None: continue
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok: continue
            b64 = base64.b64encode(jpg.tobytes()).decode()
            try:
                r = requests.post(f"{OLLAMA_HOST}/api/generate", json={
                    "model": OLLAMA_MODEL, "prompt": OLLAMA_PROMPT,
                    "images": [b64], "stream": False}, timeout=120)
                txt = r.json().get("response", "").strip()
                state.ai_text = txt or "(no response)"
                direction = self._parse_direction(txt)
                if direction:
                    state.ai_suggestion   = direction
                    state.ai_suggestion_t = time.time()
            except Exception as e:
                state.ai_text = f"(ollama error: {e})"

    @staticmethod
    def _parse_direction(txt):
        """Extract a steer direction from the model's reply. Returns
        FORWARD / LEFT / RIGHT / STOP, or None if nothing recognizable."""
        if not txt:
            return None
        head = txt.lower()
        # Prefer the explicit 'DIRECTION:' field if present
        if "direction:" in head:
            head = head.split("direction:", 1)[1]
        head = head.split("|", 1)[0]   # ignore the reason part
        for key, val in (("left", "LEFT"), ("right", "RIGHT"),
                         ("stop", "STOP"), ("forward", "FORWARD")):
            if key in head:
                return val
        return None

# ============================================================
# Navigation Thread - vision-only decisions
# ============================================================
class NavigationThread(threading.Thread):
    def __init__(self, arduino):
        super().__init__(daemon=True)
        self.arduino = arduino
        self.turn_count = 0

    def run(self):
        log.info("Navigation started.")
        while state.running and not any([
            state.obstacle_left, state.obstacle_center, state.obstacle_right]):
            time.sleep(0.1)
        while state.running:
            cmd = self._decide()
            if cmd != state.current_cmd:
                state.current_cmd = cmd
                log.info(f"-> {cmd}  L={state.obstacle_left:.0%} C={state.obstacle_center:.0%} R={state.obstacle_right:.0%}")
            self.arduino.send(cmd)
            time.sleep(DRIVE_INTERVAL)

    def _decide(self):
        # Wireless safety: if frames are stale (Wi-Fi dropout), stop.
        if not frame_is_fresh():
            return "STOP"

        if time.time() < state.avoid_until:
            return state.current_cmd
        L, C, R = state.obstacle_left, state.obstacle_center, state.obstacle_right
        close_objs = [d for d in state.detections if d.get("close")]

        if C > OBSTACLE_THRESH_STOP:
            return self._avoid()
        if close_objs:
            avg_cx = sum(d["cx"] for d in close_objs) / len(close_objs)
            if 0.3 < avg_cx < 0.7: return self._avoid()
            elif avg_cx <= 0.3:    return self._steer("RIGHT")
            else:                  return self._steer("LEFT")
        if L > OBSTACLE_THRESH_STOP and R > OBSTACLE_THRESH_STOP:
            return self._avoid()
        if L > OBSTACLE_THRESH_SLOW and L > R: return self._steer("RIGHT")
        if R > OBSTACLE_THRESH_SLOW and R > L: return self._steer("LEFT")
        if state.ir_left and state.ir_right:  return self._avoid()
        if state.ir_left:  return "RIGHT"
        if state.ir_right: return "LEFT"
        if C > OBSTACLE_THRESH_SLOW: return "SLOW_FORWARD"

        # ---- Path is clear: let the LLM (thinking brain) choose where to explore ----
        return self._ai_explore(L, R)

    def _ai_explore(self, L, R):
        """Apply the LLM's suggested direction when the way is open.
        Curves toward the suggested side in short bursts so it doesn't spin."""
        if not AI_STEER_ENABLE:
            return "FORWARD"
        now = time.time()
        if now - state.ai_suggestion_t > AI_SUGGESTION_TTL:
            return "FORWARD"                       # suggestion too old

        if now < state.ai_steer_until:             # mid-curve burst
            return state.ai_steer_dir

        sug = state.ai_suggestion
        if sug == "STOP":
            return "STOP"
        if sug in ("LEFT", "RIGHT") and now > state.ai_steer_cool:
            side_open = (L if sug == "LEFT" else R) < OBSTACLE_THRESH_SLOW
            if side_open:
                state.ai_steer_dir   = sug
                state.ai_steer_until = now + AI_TURN_BURST
                state.ai_steer_cool  = now + AI_TURN_BURST + AI_TURN_COOLDOWN
                return sug
        return "FORWARD"

    def _steer(self, direction):
        return direction

    def _avoid(self):
        self.arduino.send("BACKWARD")
        time.sleep(0.35)
        self.turn_count += 1
        turn = "LEFT" if self.turn_count % 2 == 0 else "RIGHT"
        self.arduino.send(turn)
        state.avoid_until = time.time() + 0.55
        return turn

# ============================================================
# Flask Dashboard
# ============================================================
DASH_HTML = """
<!DOCTYPE html><html><head><title>Robot Cam</title>
<meta http-equiv="refresh" content="1">
<style>
 body{background:#0d0d0d;color:#e0e0e0;font-family:monospace;margin:0;padding:16px}
 h2{margin:0 0 10px;color:#0ff;letter-spacing:2px}
 img{display:block;border:1px solid #333;width:640px;max-width:100%}
 .hud{margin-top:8px;font-size:0.95em;display:flex;gap:24px;flex-wrap:wrap}
 .hud span{background:#1a1a1a;padding:4px 10px;border-radius:4px}
 .cmd{color:#0f0;font-weight:bold}
 .warn{color:#000;background:#ff3b3b !important;font-weight:bold}
 .ai{margin-top:10px;background:#102018;border:1px solid #0a5;padding:10px;
     border-radius:6px;color:#7fffd4;max-width:640px}
 .det{margin-top:8px}
 .det span{display:inline-block;background:#1a2a1a;border:1px solid #0a0;
           padding:2px 8px;border-radius:3px;margin:2px;font-size:0.88em}
</style></head><body>
  <h2>ROBOT LIVE</h2>
  <img src="/video_feed">
  <div class="hud">
    <span>CMD: <span class="cmd">{{ cmd }}</span></span>
    <span>L: {{ ol }}</span><span>C: {{ oc }}</span><span>R: {{ or_ }}</span>
    <span class="{{ 'warn' if not cam else '' }}">CAM: {{ 'OK' if cam else 'STREAM LOST' }}</span>
  </div>
  <div class="ai"><b>AI sees:</b> {{ ai }}<br><b>AI wants:</b> {{ sug }}</div>
  <div class="det">
    {% for d in dets %}
      <span>{{ d.label }} {{ "%.0f"|format(d.conf*100) }}%{% if d.close %} (!){% endif %}</span>
    {% endfor %}
  </div>
</body></html>
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(DASH_HTML,
        cmd=state.current_cmd,
        ol=f"{state.obstacle_left:.0%}", oc=f"{state.obstacle_center:.0%}",
        or_=f"{state.obstacle_right:.0%}", ai=state.ai_text,
        sug=state.ai_suggestion, cam=frame_is_fresh(), dets=state.detections)

@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with frame_lock:
                f = state.annotated_frame
            if f is None:
                time.sleep(0.05); continue
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 72])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",   default=None)
    parser.add_argument("--no-web", action="store_true")
    parser.add_argument("--no-ai",  action="store_true")
    args = parser.parse_args()

    global OLLAMA_ENABLE
    if args.no_ai: OLLAMA_ENABLE = False

    port = args.port or find_arduino()
    if not port:
        log.error("Arduino not found! Check USB connection."); return

    arduino = ArduinoSerial(port)
    CameraThread().start()
    VisionThread().start()
    OllamaThread().start()
    NavigationThread(arduino).start()

    if not args.no_web:
        log.info(f"Dashboard -> http://0.0.0.0:{WEB_PORT}")
        threading.Thread(
            target=lambda: app.run("0.0.0.0", WEB_PORT, threaded=True, use_reloader=False),
            daemon=True).start()

    log.info("Robot running. Ctrl+C to stop.")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping...")
        state.running = False
        arduino.send("STOP")
        time.sleep(0.3)

if __name__ == "__main__":
    main()
