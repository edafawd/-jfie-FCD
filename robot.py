#!/usr/bin/env python3
"""
============================================================
Robot Car Brain - Raspberry Pi 5
- Tkinter GUI with live camera feed
- Vision-only obstacle avoidance
- YOLOv8n object detection
- Arduino serial control

Run: source ~/robot_venv/bin/activate && python3 robot_brain.py
============================================================
"""

import cv2
import serial
import serial.tools.list_ports
import threading
import time
import logging
import subprocess
import tkinter as tk
from tkinter import ttk, font
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    print("\n*** ERROR: PIL ImageTk missing. Run: bash fix_pillow.sh ***\n")
    # Try pulling ImageTk from system PIL directly as last resort
    import sys, importlib
    try:
        import importlib.util, os
        sys_pil = "/usr/lib/python3/dist-packages/PIL"
        if os.path.isdir(sys_pil):
            spec = importlib.util.spec_from_file_location(
                "PIL", os.path.join(sys_pil, "__init__.py"),
                submodule_search_locations=[sys_pil])
            pil_mod = importlib.util.module_from_spec(spec)
            sys.modules["PIL"] = pil_mod
            spec.loader.exec_module(pil_mod)
        from PIL import Image, ImageTk
        print("  âœ“ Loaded system PIL as fallback")
    except Exception as e:
        print(f"  âœ— Could not load system PIL either: {e}")
        print("  Run:  bash fix_pillow.sh")
        sys.exit(1)
from dataclasses import dataclass, field
from typing import Optional

from ultralytics import YOLO

# ============================================================
# Config
# ============================================================
SERIAL_BAUD     = 9600
FRAME_W         = 640
FRAME_H         = 480
DETECT_EVERY_N  = 3      # run YOLO every N frames (saves CPU)
CONF_THRESH     = 0.40
DRIVE_INTERVAL  = 0.15   # seconds between motor commands

OBSTACLE_ZONE_TOP = 0.50
FLOOR_SAMPLE_TOP  = 0.78
OBSTACLE_STOP     = 0.42
OBSTACLE_SLOW     = 0.22

INTERESTING = {
    0:"person", 56:"chair", 57:"couch", 59:"bed",
    60:"dining table", 62:"tv", 63:"laptop",
    67:"phone", 72:"fridge", 73:"book",
}

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("Robot")

# ============================================================
# Shared State
# ============================================================
@dataclass
class State:
    running:         bool  = True
    cmd:             str   = "STOP"
    detections:      list  = field(default_factory=list)
    obs_left:        float = 0.0
    obs_center:      float = 0.0
    obs_right:       float = 0.0
    frame_raw:       Optional[np.ndarray] = None
    frame_display:   Optional[np.ndarray] = None
    arduino_ok:      bool  = False
    camera_ok:       bool  = False
    yolo_ready:      bool  = False
    calibrated:      bool  = False
    avoid_until:     float = 0.0
    turn_count:      int   = 0
    ir_left:         int   = 0
    ir_right:        int   = 0
    manual_override: bool  = False   # True = user is driving manually
    status_msg:      str   = "Starting up..."

G = State()
frame_lock = threading.Lock()

# ============================================================
# Arduino Serial
# ============================================================
def find_arduino():
    for p in serial.tools.list_ports.comports():
        d = (p.description or "").lower()
        if any(k in d for k in ["arduino","ch340","ch341","uno"]):
            return p.device
    for dev in ["/dev/ttyUSB0","/dev/ttyACM0",
                "/dev/ttyUSB1","/dev/ttyACM1"]:
        try:
            s = serial.Serial(dev, SERIAL_BAUD, timeout=0.3)
            s.close(); return dev
        except: pass
    return None

class Arduino:
    def __init__(self):
        self.ser  = None
        self.port = None
        self._connect()
        threading.Thread(target=self._reader, daemon=True).start()

    def _connect(self):
        port = find_arduino()
        if not port:
            log.warning("Arduino not found - check USB cable")
            G.status_msg = "âš  Arduino not found - check USB"
            return
        try:
            self.ser  = serial.Serial(port, SERIAL_BAUD, timeout=1)
            self.port = port
            time.sleep(2)          # Arduino resets on connect
            G.arduino_ok  = True
            G.status_msg  = f"âœ“ Arduino on {port}"
            log.info(f"Arduino connected on {port}")
        except Exception as e:
            log.error(f"Serial error: {e}")
            G.status_msg = f"âœ— Serial error: {e}"

    def send(self, cmd: str):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((cmd.strip()+"\n").encode())
            except Exception as e:
                log.warning(f"Send failed: {e}")
        # else silently skip â€” camera still works without Arduino

    def _reader(self):
        while G.running:
            if not self.ser or not self.ser.is_open:
                time.sleep(0.5); continue
            try:
                line = self.ser.readline().decode("utf-8",errors="ignore").strip()
                if "IR_L:" in line:
                    parts = dict(p.split(":") for p in line.split(",") if ":" in p)
                    G.ir_left  = int(parts.get("IR_L",0))
                    G.ir_right = int(parts.get("IR_R",0))
                elif line == "READY":
                    log.info("Arduino READY")
            except: pass

arduino = None  # initialized in main

# ============================================================
# Camera â€” tries multiple indices and backends
# ============================================================
class Camera(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def _open(self):
        # Try indices 0,1,2 with V4L2 then default backend
        for idx in range(3):
            for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(idx, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        log.info(f"Camera opened: index={idx} backend={backend}")
                        G.camera_ok  = True
                        G.status_msg = f"âœ“ Camera on /dev/video{idx}"
                        return cap
                cap.release()
        return None

    def run(self):
        G.status_msg = "Opening camera..."
        cap = self._open()
        if cap is None:
            log.error("No camera found!")
            G.status_msg = "âœ— No camera found - check Arducam USB/CSI"
            # Show a placeholder frame so GUI doesn't crash
            placeholder = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.putText(placeholder, "NO CAMERA", (180,240),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,200), 3)
            with frame_lock:
                G.frame_raw = placeholder
            return

        while G.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                with frame_lock:
                    G.frame_raw = frame.copy()
            else:
                time.sleep(0.02)
        cap.release()

# ============================================================
# Floor Calibration + Obstacle Analyzer
# ============================================================
class ObstacleAnalyzer:
    def __init__(self):
        self.mean = None
        self.std  = None
        self.n    = 0

    def update(self, frame):
        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.n < 25:
            strip = hsv[int(h*FLOOR_SAMPLE_TOP):h, w//4:3*w//4]
            m = strip.mean(axis=(0,1))
            s = strip.std(axis=(0,1)) + 18
            if self.mean is None:
                self.mean, self.std = m, s
            else:
                a = 0.12
                self.mean = (1-a)*self.mean + a*m
                self.std  = (1-a)*self.std  + a*s
            self.n += 1
            if self.n == 25:
                G.calibrated = True
                log.info(f"Floor calibrated: HSV={self.mean.astype(int)}")
            return 0.0, 0.0, 0.0

        y1 = int(h * OBSTACLE_ZONE_TOP)
        y2 = int(h * FLOOR_SAMPLE_TOP)
        zone = hsv[y1:y2, :]

        lo = np.clip(self.mean - 2.5*self.std, 0, 255).astype(np.uint8)
        hi = np.clip(self.mean + 2.5*self.std, 0, 255).astype(np.uint8)
        floor_mask = cv2.inRange(zone, lo, hi)
        obs_mask   = cv2.bitwise_not(floor_mask)

        t = w // 3
        L = obs_mask[:, :t].mean()   / 255.0
        C = obs_mask[:, t:2*t].mean()/ 255.0
        R = obs_mask[:, 2*t:].mean() / 255.0

        G.obs_left, G.obs_center, G.obs_right = L, C, R
        return L, C, R

    def draw(self, frame):
        h, w = frame.shape[:2]
        y1, y2 = int(h*OBSTACLE_ZONE_TOP), int(h*FLOOR_SAMPLE_TOP)
        t = w // 3
        ov = frame.copy()
        for i, fill in enumerate([G.obs_left, G.obs_center, G.obs_right]):
            x1, x2 = i*t, (i+1)*t
            color = (0,0,200) if fill>OBSTACLE_STOP else (0,140,255) if fill>OBSTACLE_SLOW else (0,180,0)
            cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
            cv2.putText(frame, f"{fill:.0%}", (x1+4, y1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)
        cv2.addWeighted(ov, 0.22, frame, 0.78, 0, frame)
        cv2.rectangle(frame,(0,y1),(w,y2),(120,120,120),1)
        return frame

# ============================================================
# Vision Thread â€” YOLO + obstacle analysis
# ============================================================
class VisionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.analyzer = ObstacleAnalyzer()
        self.model    = None
        self.fc       = 0

    def _load_model(self):
        G.status_msg = "Loading YOLOv8n model..."
        try:
            self.model = YOLO("yolov8n.pt")
            G.yolo_ready = True
            G.status_msg = "âœ“ YOLOv8n ready"
            log.info("YOLOv8 loaded")
        except Exception as e:
            log.error(f"YOLO load failed: {e}")
            G.status_msg = f"âœ— YOLO error: {e}"

    def run(self):
        self._load_model()
        while G.running:
            with frame_lock:
                frame = G.frame_raw
            if frame is None:
                time.sleep(0.04); continue

            self.fc += 1
            annotated = frame.copy()
            self.analyzer.update(frame)
            self.analyzer.draw(annotated)

            dets = []
            if self.model and self.fc % DETECT_EVERY_N == 0:
                try:
                    results = self.model(frame, conf=CONF_THRESH, verbose=False)[0]
                    for box in results.boxes:
                        cid  = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1,y1,x2,y2 = map(int,box.xyxy[0])
                        label = INTERESTING.get(cid)
                        if label is None:
                            cv2.rectangle(annotated,(x1,y1),(x2,y2),(70,70,70),1)
                            continue
                        dets.append({"label":label,"conf":conf,
                                     "box":(x1,y1,x2,y2),
                                     "cx":(x1+x2)/2/FRAME_W,
                                     "close": y2 > FRAME_H*0.52})
                        col = (0,255,80) if label=="person" else (0,200,255)
                        cv2.rectangle(annotated,(x1,y1),(x2,y2),col,2)
                        cv2.putText(annotated,f"{label} {conf:.0%}",
                                    (x1,max(y1-6,12)),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.52,col,2)
                    G.detections = dets
                except Exception as e:
                    log.warning(f"YOLO error: {e}")

            # HUD
            status = "CALIBRATING..." if not G.calibrated else G.cmd
            cv2.putText(annotated,
                f"CMD:{status}  L:{G.obs_left:.0%} C:{G.obs_center:.0%} R:{G.obs_right:.0%}",
                (6, FRAME_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            with frame_lock:
                G.frame_display = annotated
            time.sleep(0.03)

# ============================================================
# Navigation Thread
# ============================================================
class NavThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)

    def run(self):
        while G.running:
            if not G.manual_override and G.calibrated:
                cmd = self._decide()
                if cmd != G.cmd:
                    G.cmd = cmd
                    log.info(f"â†’ {cmd}  L={G.obs_left:.0%} C={G.obs_center:.0%} R={G.obs_right:.0%}")
                if arduino:
                    arduino.send(cmd)
            time.sleep(DRIVE_INTERVAL)

    def _decide(self):
        if time.time() < G.avoid_until:
            return G.cmd
        L,C,R = G.obs_left, G.obs_center, G.obs_right
        close = [d for d in G.detections if d.get("close")]

        if C > OBSTACLE_STOP:                      return self._avoid()
        if close:
            cx = sum(d["cx"] for d in close)/len(close)
            if 0.3 < cx < 0.7:                    return self._avoid()
            return "RIGHT" if cx <= 0.3 else "LEFT"
        if L > OBSTACLE_STOP and R > OBSTACLE_STOP: return self._avoid()
        if L > OBSTACLE_SLOW and L > R:            return "RIGHT"
        if R > OBSTACLE_SLOW and R > L:            return "LEFT"
        if G.ir_left and G.ir_right:               return self._avoid()
        if G.ir_left:                              return "RIGHT"
        if G.ir_right:                             return "LEFT"
        if C > OBSTACLE_SLOW:                      return "SLOW_FORWARD"
        return "FORWARD"

    def _avoid(self):
        if arduino: arduino.send("BACKWARD")
        time.sleep(0.35)
        G.turn_count += 1
        t = "LEFT" if G.turn_count % 2 == 0 else "RIGHT"
        if arduino: arduino.send(t)
        G.avoid_until = time.time() + 0.55
        return t

# ============================================================
# GUI
# ============================================================
class RobotGUI:
    def __init__(self, root):
        self.root = root
        root.title("ðŸ¤– Robot Car Control Panel")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self._update_loop()

        # Keyboard controls
        root.bind("<KeyPress>",   self._key_press)
        root.bind("<KeyRelease>", self._key_release)
        root.focus_set()

    # ----------------------------------------------------------
    def _build_ui(self):
        BG    = "#1a1a2e"
        PANEL = "#16213e"
        CARD  = "#0f3460"
        ACC   = "#e94560"
        FG    = "#e0e0e0"
        GREEN = "#00b894"
        MONO  = ("Courier New", 10)

        # â”€â”€ Title bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        title_f = tk.Frame(self.root, bg=BG)
        title_f.pack(fill="x", padx=10, pady=(10,0))
        tk.Label(title_f, text="ðŸ¤–  ROBOT CAR CONTROL PANEL",
                 bg=BG, fg=ACC,
                 font=("Helvetica", 16, "bold")).pack(side="left")

        # â”€â”€ Main layout: left=camera, right=panels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        main = tk.Frame(self.root, bg=BG)
        main.pack(padx=10, pady=8)

        # Camera frame
        cam_frame = tk.Frame(main, bg=CARD, bd=2, relief="flat")
        cam_frame.grid(row=0, column=0, padx=(0,10), sticky="n")
        tk.Label(cam_frame, text=" ðŸ“· LIVE CAMERA ", bg=CARD,
                 fg=FG, font=("Helvetica",10,"bold")).pack(anchor="w", padx=6, pady=4)
        self.cam_label = tk.Label(cam_frame, bg="black",
                                  width=FRAME_W, height=FRAME_H)
        self.cam_label.pack(padx=4, pady=(0,4))

        # Right column
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="n")

        # â”€â”€ Status card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._card(right, "âš™  STATUS").pack(fill="x", pady=(0,6))
        sf = self._last_card_inner
        self.lbl_status  = self._row(sf, "System",   "...", FG, MONO)
        self.lbl_arduino = self._row(sf, "Arduino",  "...", FG, MONO)
        self.lbl_camera  = self._row(sf, "Camera",   "...", FG, MONO)
        self.lbl_yolo    = self._row(sf, "YOLOv8",   "...", FG, MONO)
        self.lbl_calib   = self._row(sf, "Floor Cal","...", FG, MONO)

        # â”€â”€ Drive state card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._card(right, "ðŸš—  DRIVE STATE").pack(fill="x", pady=(0,6))
        df = self._last_card_inner
        self.lbl_cmd   = self._row(df, "Command", "STOP", ACC, ("Courier New",14,"bold"))
        self.lbl_mode  = self._row(df, "Mode",    "AUTO", FG, MONO)

        # Obstacle bars
        bar_f = tk.Frame(df, bg=PANEL)
        bar_f.pack(fill="x", pady=4)
        tk.Label(bar_f, text="Obstacle zones:", bg=PANEL, fg=FG,
                 font=MONO).pack(anchor="w")
        bars = tk.Frame(bar_f, bg=PANEL)
        bars.pack()
        self.bar_l = self._bar(bars, "LEFT",   "#e74c3c"); self.bar_l[0].pack(side="left",padx=4)
        self.bar_c = self._bar(bars, "CENTER", "#e74c3c"); self.bar_c[0].pack(side="left",padx=4)
        self.bar_r = self._bar(bars, "RIGHT",  "#e74c3c"); self.bar_r[0].pack(side="left",padx=4)

        # â”€â”€ Manual control â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._card(right, "ðŸ•¹  MANUAL CONTROL  (keyboard: W A S D / arrows)").pack(fill="x", pady=(0,6))
        mf = self._last_card_inner

        btn_style = dict(bg=CARD, fg=FG, font=("Helvetica",11,"bold"),
                         relief="flat", bd=0, padx=14, pady=8,
                         activebackground=ACC, activeforeground="white",
                         cursor="hand2")

        grid = tk.Frame(mf, bg=PANEL)
        grid.pack(pady=4)
        self.btn_fwd  = tk.Button(grid, text="â–²  FORWARD",  **btn_style,
                                  command=lambda:self._manual("FORWARD"))
        self.btn_back = tk.Button(grid, text="â–¼  BACKWARD", **btn_style,
                                  command=lambda:self._manual("BACKWARD"))
        self.btn_left = tk.Button(grid, text="â—€  LEFT",     **btn_style,
                                  command=lambda:self._manual("LEFT"))
        self.btn_right= tk.Button(grid, text="RIGHT  â–¶",    **btn_style,
                                  command=lambda:self._manual("RIGHT"))
        stop_style = {**btn_style, "bg": "#7f0000"}
        self.btn_stop = tk.Button(grid, text="â¬›  STOP",
                                  command=lambda:self._manual("STOP"),
                                  **stop_style)

        self.btn_fwd.grid  (row=0,column=1,pady=2,padx=2)
        self.btn_left.grid (row=1,column=0,pady=2,padx=2)
        self.btn_stop.grid (row=1,column=1,pady=2,padx=2)
        self.btn_right.grid(row=1,column=2,pady=2,padx=2)
        self.btn_back.grid (row=2,column=1,pady=2,padx=2)

        # Auto/Manual toggle
        self.auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(mf, text="  Manual override (disables auto-drive)",
                       variable=self.auto_var,
                       command=self._toggle_mode,
                       bg=PANEL, fg=FG, selectcolor=CARD,
                       activebackground=PANEL, activeforeground=FG,
                       font=MONO).pack(anchor="w", pady=2)

        # â”€â”€ Detections list â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._card(right, "ðŸ”  DETECTED OBJECTS").pack(fill="x", pady=(0,6))
        det_f = self._last_card_inner
        self.det_list = tk.Text(det_f, bg="#0a0a1a", fg=GREEN,
                                font=MONO, height=6, width=32,
                                relief="flat", state="disabled",
                                insertbackground=GREEN)
        self.det_list.pack(fill="x")

        # â”€â”€ Log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        self._card(right, "ðŸ“‹  LOG").pack(fill="x")
        log_f = self._last_card_inner
        self.log_box = tk.Text(log_f, bg="#0a0a1a", fg="#888",
                               font=("Courier New",9), height=5,
                               width=32, relief="flat", state="disabled")
        self.log_box.pack(fill="x")

        # Status bar at bottom
        self.status_bar = tk.Label(self.root,
            text="Initializing...", bg="#0a0a1a", fg="#666",
            font=("Courier New",9), anchor="w")
        self.status_bar.pack(fill="x", padx=10, pady=(0,6))

    # â”€â”€ UI helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _card(self, parent, title):
        PANEL="#16213e"; CARD="#0f3460"; FG="#e0e0e0"
        outer = tk.Frame(parent, bg=CARD, bd=1, relief="flat")
        tk.Label(outer, text=f" {title} ", bg=CARD, fg="#aaa",
                 font=("Helvetica",9,"bold")).pack(anchor="w",padx=4,pady=(4,0))
        inner = tk.Frame(outer, bg=PANEL, padx=8, pady=6)
        inner.pack(fill="x", padx=4, pady=(2,4))
        self._last_card_inner = inner
        return outer

    def _row(self, parent, key, val, fg, fnt):
        f = tk.Frame(parent, bg=parent["bg"])
        f.pack(fill="x", pady=1)
        tk.Label(f, text=f"{key}:", width=10, anchor="w",
                 bg=parent["bg"], fg="#666",
                 font=("Courier New",9)).pack(side="left")
        lbl = tk.Label(f, text=val, anchor="w",
                       bg=parent["bg"], fg=fg, font=fnt)
        lbl.pack(side="left")
        return lbl

    def _bar(self, parent, label, color):
        BG="#16213e"
        f = tk.Frame(parent, bg=BG)
        tk.Label(f, text=label, bg=BG, fg="#aaa",
                 font=("Courier New",8)).pack()
        canvas = tk.Canvas(f, width=40, height=80,
                           bg="#0a0a1a", highlightthickness=0)
        canvas.pack()
        rect = canvas.create_rectangle(0,80,40,80, fill=color, outline="")
        pct  = tk.Label(f, text="0%", bg=BG, fg="#aaa",
                        font=("Courier New",8))
        pct.pack()
        return f, canvas, rect, pct

    def _update_bar(self, bar_tuple, value):
        _, canvas, rect, pct = bar_tuple
        h = int(80 * value)
        canvas.coords(rect, 0, 80-h, 40, 80)
        color = "#e74c3c" if value>OBSTACLE_STOP else "#f39c12" if value>OBSTACLE_SLOW else "#2ecc71"
        canvas.itemconfig(rect, fill=color)
        pct.config(text=f"{value:.0%}")

    # â”€â”€ Live update loop (runs every 40ms in Tk main thread) â”€â”€â”€
    def _update_loop(self):
        # Camera
        with frame_lock:
            frame = G.frame_display or G.frame_raw
        if frame is not None:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img  = Image.fromarray(rgb)
            img  = img.resize((FRAME_W, FRAME_H), Image.NEAREST)
            imtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imtk   # keep ref!
            self.cam_label.config(image=imtk)

        # Status
        ok = lambda b: ("âœ“","#00b894") if b else ("âœ—","#e74c3c")
        sym,col = ok(G.arduino_ok);  self.lbl_arduino.config(text=sym,fg=col)
        sym,col = ok(G.camera_ok);   self.lbl_camera.config(text=sym,fg=col)
        sym,col = ok(G.yolo_ready);  self.lbl_yolo.config(text=sym,fg=col)
        sym,col = ok(G.calibrated);  self.lbl_calib.config(text=sym,fg=col)
        self.lbl_status.config(text=G.status_msg[:28])
        self.lbl_cmd.config(text=G.cmd)
        self.lbl_mode.config(text="MANUAL" if G.manual_override else "AUTO")

        # Obstacle bars
        self._update_bar(self.bar_l, G.obs_left)
        self._update_bar(self.bar_c, G.obs_center)
        self._update_bar(self.bar_r, G.obs_right)

        # Detections
        self.det_list.config(state="normal")
        self.det_list.delete("1.0","end")
        if G.detections:
            for d in G.detections:
                close = " âš  CLOSE" if d.get("close") else ""
                self.det_list.insert("end",
                    f"  {d['label']:15s} {d['conf']:.0%}{close}\n")
        else:
            self.det_list.insert("end","  (nothing detected)")
        self.det_list.config(state="disabled")

        # Status bar
        self.status_bar.config(text=f"  {G.status_msg}")

        self.root.after(40, self._update_loop)   # ~25 fps UI refresh

    # â”€â”€ Manual control â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _manual(self, cmd):
        if G.manual_override and arduino:
            G.cmd = cmd
            arduino.send(cmd)

    def _toggle_mode(self):
        G.manual_override = self.auto_var.get()
        if not G.manual_override and arduino:
            arduino.send("STOP")

    def _key_press(self, e):
        if not G.manual_override: return
        k = e.keysym.lower()
        m = {"w":"FORWARD","s":"BACKWARD","a":"LEFT","d":"RIGHT",
             "up":"FORWARD","down":"BACKWARD","left":"LEFT","right":"RIGHT",
             "space":"STOP"}
        cmd = m.get(k)
        if cmd: self._manual(cmd)

    def _key_release(self, e):
        if not G.manual_override: return
        k = e.keysym.lower()
        if k in ("w","s","a","d","up","down","left","right"):
            self._manual("STOP")

    def _log(self, msg):
        self.log_box.config(state="normal")
        self.log_box.insert("end", msg+"\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def on_close(self):
        G.running = False
        if arduino: arduino.send("STOP")
        time.sleep(0.2)
        self.root.destroy()

# ============================================================
# Main
# ============================================================
def main():
    global arduino

    # Start background threads before GUI
    arduino = Arduino()
    Camera().start()
    VisionThread().start()
    NavThread().start()

    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
