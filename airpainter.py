# airpainter_ultra.py
"""
AirPainter ULTRA — single-file (high-accuracy)
Requirements: Python 3.10, opencv-python, mediapipe, numpy, pillow
Optional: SpeechRecognition + pyaudio (voice commands), winsound works on Windows for sound.
"""

import os
import time
import threading
import math
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Optional voice libs
VOICE_AVAILABLE = False
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# Optional winsound for feedback on Windows
try:
    import winsound

    def play_beep(freq=800, dur=120):
        winsound.Beep(freq, dur)
except Exception:
    def play_beep(freq=800, dur=120):
        pass  # no-op if winsound not available

# ===== Settings =====
CAM_W, CAM_H = 640, 480
SAVE_DIR = "saved_drawings"
os.makedirs(SAVE_DIR, exist_ok=True)

TIMELAPSE_FPS = 15
TIMELAPSE_FRAME_INTERVAL = 2

# Smoothing params
INTERP_BASE = 4
DEADZONE_PIX = 2

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_seg = mp.solutions.selfie_segmentation

# ==== Utility classes ====

class LandmarkStabilizer:
    """Median window + EMA per-landmark stabilizer."""
    def __init__(self, num_landmarks=21, median_k=5, ema_alpha=0.6):
        self.k = median_k if median_k % 2 == 1 else median_k + 1
        self.ema_alpha = ema_alpha
        self.buffers = [deque(maxlen=self.k) for _ in range(num_landmarks)]
        self.ema = [None] * num_landmarks

    def update(self, landmarks):
        out = []
        for i, p in enumerate(landmarks):
            self.buffers[i].append(p)
            xs = [v[0] for v in self.buffers[i]]
            ys = [v[1] for v in self.buffers[i]]
            zs = [v[2] for v in self.buffers[i]]
            # median
            mx = int(sorted(xs)[len(xs)//2])
            my = int(sorted(ys)[len(ys)//2])
            mz = float(sorted(zs)[len(zs)//2])
            median_pt = (mx, my, mz)
            if self.ema[i] is None:
                self.ema[i] = median_pt
            else:
                ex, ey, ez = self.ema[i]
                a = self.ema_alpha
                self.ema[i] = (int(ex*(1-a) + mx*a),
                               int(ey*(1-a) + my*a),
                               ez*(1-a) + mz*a)
            out.append(self.ema[i])
        return out

class SimpleKalman:
    """Tiny 2D Kalman for fingertip smoothing."""
    def __init__(self, q=1e-2, r=1e-1):
        self.q = q; self.r = r
        self.P = np.eye(4) * 1.0
        self.x = np.zeros((4,1))
        self._init = False

    def predict(self, dt=1.0):
        F = np.array([[1, dt, 0,  0],
                      [0,  1, 0,  0],
                      [0,  0, 1, dt],
                      [0,  0, 0,  1]])
        Q = np.eye(4) * self.q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([[1,0,0,0],
                      [0,0,1,0]])
        R = np.eye(2) * self.r
        y = np.array(z).reshape(2,1) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def step(self, meas):
        if not self._init:
            self.x = np.array([[meas[0]],[0],[meas[1]],[0]], dtype=float)
            self._init = True
            return (int(meas[0]), int(meas[1]))
        self.predict(dt=1.0)
        self.update(meas)
        return (int(self.x[0,0]), int(self.x[2,0]))


# ==== Drawing engine ====
class DrawingEngine:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.prev_point = None
        self.brush_color = (0, 0, 255)  # BGR
        self.brush_size = 7
        self.eraser = False
        self.history = []
        self.redo_stack = []
        self.bg_image = None
        self.seg_mask = None
        self.record_frames = []
        self.frame_count = 0
        self.current_stroke = []

    def save_state(self):
        self.history.append(self.canvas.copy())
        if len(self.history) > 40:
            self.history.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if not self.history: return
        self.redo_stack.append(self.canvas.copy())
        self.canvas = self.history.pop()

    def redo(self):
        if not self.redo_stack: return
        self.history.append(self.canvas.copy())
        self.canvas = self.redo_stack.pop()

    def clear(self):
        self.save_state()
        self.canvas[:] = 0

    def load_bg(self, path):
        img = cv2.imread(path)
        if img is None: return
        img = cv2.resize(img, (self.w, self.h))
        self.bg_image = img

    def set_mask(self, mask):
        if mask is None:
            self.seg_mask = None
            return
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32)
        self.seg_mask = cv2.resize(mask, (self.w, self.h))

    def _apply_point(self, p):
        x,y = p
        color = (0,0,0) if self.eraser else self.brush_color
        size = max(1, (40 if self.eraser else self.brush_size))
        cv2.circle(self.canvas, (x,y), size//2, color, -1)

    def draw_line(self, p1, p2):
        color = (0,0,0) if self.eraser else self.brush_color
        size = max(1, (40 if self.eraser else self.brush_size))
        if self.seg_mask is None:
            cv2.line(self.canvas, p1, p2, color, size)
        else:
            tmp = np.zeros_like(self.canvas)
            cv2.line(tmp, p1, p2, color, size)
            mask3 = np.repeat((self.seg_mask>0.5)[:,:,None], 3, axis=2).astype(np.uint8)
            self.canvas = np.where(mask3, cv2.add(self.canvas, tmp), self.canvas)

    def draw(self, point):
        # point: (x,y) or None to lift pen
        if point is None:
            self.prev_point = None
            if self.current_stroke:
                # keep stroke for possible shape
                pass
            return

        x,y = point
        self.current_stroke.append((x,y))
        if self.prev_point is None:
            self.prev_point = (x,y)
            self._apply_point((x,y))
            return

        x0,y0 = self.prev_point
        dx = x - x0; dy = y - y0
        if (dx*dx + dy*dy) < (DEADZONE_PIX*DEADZONE_PIX):
            # ignore micro jitter
            return

        speed = math.hypot(dx,dy)
        steps = int(max(2, min(12, INTERP_BASE + speed//20)))
        for t in np.linspace(0,1,steps):
            xi = int(x0 + (x-x0)*t)
            yi = int(y0 + (y-y0)*t)
            self.draw_line((xi,yi),(xi,yi))
        self.draw_line((x0,y0),(x,y))
        self.prev_point = (x,y)

    def convert_stroke_to_shape(self, shape_type='auto'):
        pts = self.current_stroke.copy()
        if len(pts) < 5:
            self.current_stroke = []
            return
        arr = np.array(pts)
        self.save_state()
        if shape_type == 'auto':
            x_min, y_min = arr.min(axis=0)
            x_max, y_max = arr.max(axis=0)
            w = x_max - x_min; h = y_max - y_min
            ratio = min(w,h)/(max(w,h)+1e-6)
            if ratio > 0.8:
                shape = 'circle'
            elif min(w,h) < 20 or max(w,h)/(min(w,h)+1e-6) > 5:
                shape = 'line'
            else:
                shape = 'rect'
        else:
            shape = shape_type

        color = (0,0,0) if self.eraser else self.brush_color

        if shape == 'circle':
            (cx,cy), r = cv2.minEnclosingCircle(arr.astype(np.int32))
            center = (int(cx), int(cy)); radius = int(r)
            if self.seg_mask is None:
                cv2.circle(self.canvas, center, radius, color, max(2, self.brush_size))
            else:
                tmp = np.zeros_like(self.canvas)
                cv2.circle(tmp, center, radius, color, max(2, self.brush_size))
                mask3 = np.repeat((self.seg_mask>0.5)[:,:,None],3,axis=2).astype(np.uint8)
                self.canvas = np.where(mask3, cv2.add(self.canvas, tmp), self.canvas)

        elif shape == 'rect':
            rect = cv2.minAreaRect(arr.astype(np.int32))
            box = cv2.boxPoints(rect).astype(int)
            if self.seg_mask is None:
                cv2.drawContours(self.canvas, [box], 0, color, max(2, self.brush_size))
            else:
                tmp = np.zeros_like(self.canvas)
                cv2.drawContours(tmp, [box], 0, color, max(2, self.brush_size))
                mask3 = np.repeat((self.seg_mask>0.5)[:,:,None],3,axis=2).astype(np.uint8)
                self.canvas = np.where(mask3, cv2.add(self.canvas, tmp), self.canvas)
        else:  # line
            vx, vy, x0, y0 = cv2.fitLine(arr.astype(np.int32), cv2.DIST_L2, 0,0.01,0.01)
            lefty = int(((0 - x0) * vy / vx) + y0) if abs(vx)>1e-6 else 0
            righty = int(((self.w - x0) * vy / vx) + y0) if abs(vx)>1e-6 else self.h
            if self.seg_mask is None:
                cv2.line(self.canvas, (0,lefty), (self.w, righty), color, max(2, self.brush_size))
            else:
                tmp = np.zeros_like(self.canvas)
                cv2.line(tmp, (0,lefty), (self.w, righty), color, max(2, self.brush_size))
                mask3 = np.repeat((self.seg_mask>0.5)[:,:,None],3,axis=2).astype(np.uint8)
                self.canvas = np.where(mask3, cv2.add(self.canvas, tmp), self.canvas)

        self.current_stroke = []

    def get_final_output(self, frame):
        # blend with background optionally
        if self.bg_image is not None:
            blended = cv2.addWeighted(self.bg_image, 0.4, frame, 0.6, 0)
        else:
            blended = frame.copy()
        out = cv2.addWeighted(blended, 0.7, self.canvas, 0.3, 0)
        self.frame_count += 1
        if self.frame_count % TIMELAPSE_FRAME_INTERVAL == 0:
            self.record_frames.append(out.copy())
            if len(self.record_frames) > TIMELAPSE_FPS * 60:
                self.record_frames.pop(0)
        return out

    def save_image_transparent(self, path):
        # create alpha mask from canvas (non-black -> alpha 255)
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 4, 255, cv2.THRESH_BINARY)
        b,g,r = cv2.split(self.canvas)
        rgba = cv2.merge([b,g,r,alpha])
        cv2.imwrite(path, rgba)
        return path

    def save(self):
        fname = os.path.join(SAVE_DIR, f"airpaint_{int(time.time())}.png")
        cv2.imwrite(fname, self.canvas)
        return fname

    def save_timelapse(self):
        if not self.record_frames:
            return None
        fname = os.path.join(SAVE_DIR, f"timelapse_{int(time.time())}.mp4")
        h,w,_ = self.record_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(fname, fourcc, TIMELAPSE_FPS, (w,h))
        for f in self.record_frames:
            writer.write(f)
        writer.release()
        return fname

# ==== Hand Tracker (with smoothing) ====
class HandTracker:
    def __init__(self, seg_enabled=True, low_cpu=False, max_hands=2):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.seg_enabled = seg_enabled
        self.low_cpu = low_cpu
        self.max_hands = max_hands
        self.hands = mp_hands.Hands(max_num_hands=max_hands,
                                    min_detection_confidence=0.75 if not low_cpu else 0.6,
                                    min_tracking_confidence=0.7 if not low_cpu else 0.6)
        self.seg = mp_seg.SelfieSegmentation(model_selection=1) if seg_enabled else None
        self.stabilizers = []
        self.kalmans = []

    def _hand_size(self, lms):
        xs = [p[0] for p in lms]
        ys = [p[1] for p in lms]
        bbox_w = max(xs) - min(xs)
        bbox_h = max(ys) - min(ys)
        return max(1.0, (bbox_w + bbox_h) / 2.0)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, (None,[])
        frame = cv2.flip(frame, 1)
        h,w,_ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_res = self.hands.process(rgb)
        seg_mask = None
        if self.seg is not None:
            seg_res = self.seg.process(rgb)
            if seg_res.segmentation_mask is not None:
                seg_mask = seg_res.segmentation_mask  # float32

        draw_point = None
        hand_boxes = []  # list of ((x1,y1,x2,y2), (ix,iy), gesture, hand_size)

        if hands_res.multi_hand_landmarks:
            # ensure stabilizers/kalmans
            while len(self.stabilizers) < len(hands_res.multi_hand_landmarks):
                self.stabilizers.append(LandmarkStabilizer())
            while len(self.kalmans) < len(hands_res.multi_hand_landmarks):
                self.kalmans.append(SimpleKalman())

            for i, hand_lms in enumerate(hands_res.multi_hand_landmarks):
                lm = hand_lms.landmark
                landmarks = []
                for p in lm:
                    px = int(p.x * w); py = int(p.y * h); pz = p.z
                    landmarks.append((px, py, pz))

                # smooth landmarks
                sm = self.stabilizers[i].update(landmarks)
                ix, iy, _ = sm[8]
                tx, ty, _ = sm[4]
                # kalman optional:
                kx, ky = self.kalmans[i].step((ix,iy))
                ix, iy = kx, ky

                # finger up detection
                index_up = sm[8][1] < sm[6][1]
                middle_up = sm[12][1] < sm[10][1]
                # adapt thresholds by hand size
                hand_size = self._hand_size(sm)
                pinch_thresh = max(12, int(hand_size * 0.12))
                thumb_index_dist = int(math.hypot(ix - tx, iy - ty))
                pinch = thumb_index_dist < pinch_thresh

                gesture = "none"
                if index_up and not middle_up:
                    gesture = "draw"
                elif index_up and middle_up:
                    gesture = "select"
                elif pinch:
                    gesture = "pinch"

                xs = [p[0] for p in sm]; ys = [p[1] for p in sm]
                x1,y1,x2,y2 = min(xs), min(ys), max(xs), max(ys)
                hand_boxes.append(((x1,y1,x2,y2), (ix,iy), gesture, hand_size))

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # choose primary hand for drawing (right-most by centre x)
            if hand_boxes:
                # sort by center x descending (right-most first) — user preference
                hand_boxes_sorted = sorted(hand_boxes, key=lambda hb: ((hb[0][0]+hb[0][2])//2), reverse=True)
                # primary is first
                draw_point = hand_boxes_sorted[0][1]  # (ix,iy)
                # but deliver full hand_boxes for tap handling
                return frame, seg_mask, (draw_point, hand_boxes)
        return frame, seg_mask, (None, hand_boxes)

    def stop(self):
        try:
            self.cap.release()
        except:
            pass

# ==== Voice Controller ====
class VoiceController(threading.Thread):
    def __init__(self, callback):
        super().__init__(daemon=True)
        self.callback = callback
        self._stop = False
        if VOICE_AVAILABLE:
            self.rec = sr.Recognizer()
            try:
                self.mic = sr.Microphone()
            except Exception:
                self.rec = None; self.mic = None
        else:
            self.rec = None; self.mic = None

    def run(self):
        if self.mic is None or self.rec is None:
            return
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source, duration=1)
            while not self._stop:
                try:
                    audio = self.rec.listen(source, phrase_time_limit=3)
                    txt = self.rec.recognize_google(audio).lower()
                    self.callback(txt)
                except Exception:
                    continue

    def stop(self):
        self._stop = True

# ==== Tkinter App ====
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AirPainter ULTRA")
        self.geometry(f"{CAM_W+420}x{CAM_H+60}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.tracker = HandTracker(seg_enabled=True, low_cpu=False, max_hands=2)
        self.engine = DrawingEngine(CAM_W, CAM_H)
        self._stop = False

        # toolbar regions (top-left)
        self.toolbar_areas = [
            ("Color", (20,10,80,70)),
            ("Brush", (90,10,150,70)),
            ("Eraser", (160,10,220,70)),
            ("Shape", (230,10,290,70)),
            ("Undo", (300,10,360,70)),
            ("Redo", (370,10,430,70)),
            ("Clear", (440,10,500,70)),
            ("Save", (510,10,570,70)),
            ("Timelapse", (580,10,640,70)),
        ]

        # UI
        self.video_label = ttk.Label(self)
        self.video_label.pack(side=tk.LEFT, padx=2, pady=2)

        panel = ttk.Frame(self)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        ttk.Label(panel, text="AirPainter ULTRA", font=("Arial", 14, "bold")).pack(pady=6)

        # color chooser
        ttk.Button(panel, text="Choose Color", command=self.choose_color).pack(fill=tk.X, pady=4)
        self.col_preview = ttk.Label(panel, text="      ", background="#0000FF")
        self.col_preview.pack(pady=4)

        ttk.Label(panel, text="Brush Size").pack(pady=4)
        self.br_slider = ttk.Scale(panel, from_=1, to=70, command=self.on_brush_change)
        self.br_slider.set(self.engine.brush_size)
        self.br_slider.pack(fill=tk.X, pady=4)

        self.eraser_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Eraser", variable=self.eraser_var, command=self.toggle_eraser).pack(pady=4)

        self.shape_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="Shape Assist (auto)", variable=self.shape_var).pack(pady=4)

        self.seg_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(panel, text="AI Segmentation", variable=self.seg_var, command=self.toggle_seg).pack(pady=4)

        ttk.Button(panel, text="Load Background", command=self.load_background).pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Save PNG (transparent)", command=self.save_transparent).pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Save Timelapse", command=self.save_timelapse).pack(fill=tk.X, pady=4)
        ttk.Button(panel, text="Clear Canvas", command=self.engine.clear).pack(fill=tk.X, pady=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.status_var, foreground="blue").pack(pady=6)

        # voice
        self.voice_controller = None
        if VOICE_AVAILABLE:
            try:
                self.voice_controller = VoiceController(self.process_voice_command)
                self.voice_controller.start()
            except Exception:
                self.voice_controller = None

        # start camera thread
        threading.Thread(target=self.camera_loop, daemon=True).start()

    # UI callbacks
    def choose_color(self):
        col = colorchooser.askcolor()[0]
        if col:
            bgr = (int(col[2]), int(col[1]), int(col[0]))
            self.engine.brush_color = bgr
            hexc = '#%02x%02x%02x' % (int(col[0]), int(col[1]), int(col[2]))
            self.col_preview.configure(background=hexc)
            self.eraser_var.set(False)
            play_beep(1000, 80)
            self.status_var.set("Color changed")

    def on_brush_change(self, v):
        try:
            self.engine.brush_size = int(float(v))
        except:
            pass

    def toggle_eraser(self):
        self.engine.eraser = self.eraser_var.get()
        if self.engine.eraser:
            self.status_var.set("Eraser on")
            play_beep(600, 90)
        else:
            self.status_var.set("Eraser off")

    def toggle_seg(self):
        on = self.seg_var.get()
        # restart tracker with new seg option
        self.tracker.stop()
        self.tracker = HandTracker(seg_enabled=on, low_cpu=self.tracker.low_cpu, max_hands=2)
        self.status_var.set(f"Segmentation {'on' if on else 'off'}")

    def load_background(self):
        p = filedialog.askopenfilename()
        if p:
            self.engine.load_bg(p)
            self.status_var.set("Background loaded")

    def save_transparent(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")])
        if not path:
            return
        self.engine.save_image_transparent(path)
        self.status_var.set(f"Saved transparent PNG to {path}")
        play_beep(900, 120)

    def save_timelapse(self):
        p = self.engine.save_timelapse()
        if p:
            messagebox.showinfo("Saved", f"Timelapse saved: {p}")
            self.status_var.set("Timelapse saved")
            play_beep(1200, 120)
        else:
            messagebox.showinfo("No frames", "No timelapse frames recorded yet")
            self.status_var.set("No frames to save")

    def process_voice_command(self, txt):
        txt = txt.lower()
        if "clear" in txt:
            self.engine.clear(); self.status_var.set("Voice: clear"); play_beep(500,80)
        elif "save" in txt:
            p = self.engine.save(); self.status_var.set(f"Voice: saved {p}"); play_beep(1000,80)
        elif "timelapse" in txt:
            p = self.engine.save_timelapse(); self.status_var.set(f"Voice: timelapse {p}"); play_beep(1200,80)
        elif "erase" in txt:
            self.engine.eraser = True; self.eraser_var.set(True); self.status_var.set("Voice: eraser on"); play_beep(600,80)
        elif "color" in txt:
            if "red" in txt: self.engine.brush_color=(0,0,255); self.status_var.set("Voice: red"); play_beep(1000,80)
            if "green" in txt: self.engine.brush_color=(0,255,0); self.status_var.set("Voice: green"); play_beep(1000,80)
            if "blue" in txt: self.engine.brush_color=(255,0,0); self.status_var.set("Voice: blue"); play_beep(1000,80)

    def handle_toolbar_tap(self, x, y):
        for name, (x1,y1,x2,y2) in self.toolbar_areas:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "Color":
                    self.choose_color()
                elif name == "Brush":
                    self.engine.brush_size = min(70, self.engine.brush_size + 4)
                    self.br_slider.set(self.engine.brush_size)
                    self.status_var.set(f"Brush size {self.engine.brush_size}")
                elif name == "Eraser":
                    self.engine.eraser = True; self.eraser_var.set(True); self.status_var.set("Toolbar eraser"); play_beep(600,80)
                elif name == "Shape":
                    self.engine.convert_stroke_to_shape('auto'); self.status_var.set("Shape converted"); play_beep(900,80)
                elif name == "Undo":
                    self.engine.undo(); self.status_var.set("Undo")
                elif name == "Redo":
                    self.engine.redo(); self.status_var.set("Redo")
                elif name == "Clear":
                    self.engine.clear(); self.status_var.set("Cleared"); play_beep(500,80)
                elif name == "Save":
                    p = self.engine.save(); self.status_var.set(f"Saved {p}"); play_beep(1000,120)
                elif name == "Timelapse":
                    p = self.engine.save_timelapse()
                    if p: self.status_var.set(f"Timelapse {p}"); play_beep(1200,120)
                    else: self.status_var.set("No timelapse frames")
                break

    def camera_loop(self):
        last_drawing = False
        idle_frames = 0
        last_ts = time.time()
        while not self._stop:
            frame, seg_mask, info = self.tracker.get_frame()
            if frame is None:
                continue
            draw_point, hand_boxes = info

            # update segmentation mask if enabled
            if seg_mask is not None and self.seg_var.get():
                self.engine.set_mask(seg_mask)
            else:
                self.engine.set_mask(None)

            tapped = False
            # handle pinch taps (toolbar)
            for (box, idx_pt, gesture, hsize) in hand_boxes:
                if gesture == "pinch":
                    x,y = idx_pt
                    if y < 80:  # toolbar region
                        self.handle_toolbar_tap(x,y)
                        tapped = True
                        time.sleep(0.22)
                        break

            # draw or idle
            if draw_point is not None and not tapped:
                self.engine.draw(draw_point)
                last_drawing = True
                idle_frames = 0
            else:
                idle_frames += 1
                if last_drawing and idle_frames > 8:
                    if self.shape_var.get():
                        self.engine.convert_stroke_to_shape('auto')
                        play_beep(900,80)
                    last_drawing = False

            # overlay toolbar
            for name, (x1,y1,x2,y2) in self.toolbar_areas:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (60,60,60), -1)
                cv2.putText(frame, name, (x1+4, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            out = self.engine.get_final_output(frame)
            # fps
            now = time.time(); dt = now - last_ts if now != last_ts else 0.001; last_ts = now
            fps = int(1.0/dt) if dt>0 else 0
            cv2.putText(out, f"FPS: {fps}", (10, CAM_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # segmentation preview
            if self.engine.seg_mask is not None:
                mask_vis = (self.engine.seg_mask*255).astype(np.uint8)
                mask_vis = cv2.resize(mask_vis, (160,120))
                mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
                out[10:10+120, CAM_W-160:CAM_W] = mask_vis

            # show image in tkinter
            img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(img)
            self.after(0, lambda im=imgtk: self.update_image(im))

        self.tracker.stop()

    def update_image(self, imgtk):
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def on_close(self):
        self._stop = True
        if self.voice_controller: self.voice_controller.stop()
        self.tracker.stop()
        self.destroy()

if __name__ == "__main__":
    App().mainloop()
