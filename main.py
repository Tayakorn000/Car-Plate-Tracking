import os
import cv2
import csv
import time
import threading
import numpy as np
from datetime import datetime
from queue import Queue
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import re
import easyocr
import torch

try:
    from ultralytics import YOLO
    _yolo_available = True
except:
    _yolo_available = False
    print("Warning: YOLOv11 not found. Install: pip install ultralytics")

try:
    _ocr_reader = easyocr.Reader(['en'], gpu=False)
except:
    _ocr_reader = None

try:
    _gpu_available = torch.cuda.is_available()
except:
    _gpu_available = False

CAPTURE_DIR = r"d:\Car plate tracking\captures"
CSV_PATH = r"d:\Car plate tracking\detections.csv"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Helper to download YOLO model if missing
def ensure_yolo_model(model_name="yolov10n"):
    """Try multiple ways to load YOLO model (file .pt, model name string)."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"ultralytics import failed: {e}")
        return None

    candidates = [f"{model_name}.pt", model_name, f"{model_name}.yaml"]
    for c in candidates:
        try:
            print(f"Trying to load YOLO model: {c}")
            model = YOLO(c)
            print(f"Loaded YOLO model: {c}")
            return model
        except Exception as e:
            print(f"Failed to load {c}: {e}")
    print("No YOLO model loaded.")
    return None

# Enhanced image for clarity
def enhance_image(img, target_width=800):
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enh = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    try:
        enh = cv2.bilateralFilter(enh, 9, 75, 75)
    except:
        pass
    try:
        enh = cv2.fastNlMeansDenoisingColored(enh, None, 12, 12, 7, 21)
    except:
        pass
    blur = cv2.GaussianBlur(enh, (0, 0), 1.5)
    enh = cv2.addWeighted(enh, 2.5, blur, -1.5, 0)
    if w < target_width:
        scale = target_width / float(w)
        enh = cv2.resize(enh, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    return enh

# Detect plate region (OpenALPR-like morphology)
def detect_plate_region(img, min_width=50, max_width=None):
    if img is None or img.size == 0:
        return img
    try:
        h, w = img.shape[:2]
        if max_width is None:
            max_width = w
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        w = max(1, img.shape[1])
        h = max(1, img.shape[0])
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(w*0.3)), 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, int(h*0.2))))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_v, iterations=1)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_plate = None
        best_score = -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_width * 20:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = float(cw) / max(1, ch)
            if aspect < 2.0 or aspect > 10.0:
                continue
            if cw < min_width or cw > max_width or ch < 15:
                continue
            pos_score = 1.0 if y > h * 0.5 else 0.5
            size_score = min(cw, max_width) / float(max_width)
            score = pos_score * size_score
            if score > best_score:
                best_score = score
                best_plate = (x, y, cw, ch)
        if best_plate:
            x, y, cw, ch = best_plate
            pad_x = max(0, int(cw * 0.05))
            pad_y = max(0, int(ch * 0.1))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)
            return img[y1:y2, x1:x2].copy()
    except:
        pass
    return img

# OCR plate text
def ocr_plate(img):
    if img is None or getattr(img, "size", 0) == 0:
        return ""

    try:
        plate = detect_plate_region(img)
        if plate is None or plate.size == 0:
            return ""

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        if _ocr_reader:
            res = _ocr_reader.readtext(gray)
            texts = [t[1] for t in res if t[2] > 0.3]

            if texts:
                text = " ".join(texts)
                text = "".join(c for c in text if c.isalnum() or c == " ")
                text = " ".join(text.split())

                # ต้องมีอย่างน้อย 4 ตัวอักษร
                if len(text) >= 4 and sum(1 for c in text if c.isalnum()) >= 4:
                    return text

    except:
        return ""

    return ""

def clean_plate_text(self, text):
    if not text:
        return None
    text = text.replace("I", "1").replace("|", "1")
    text = text.replace(" ", "")
    pattern = r"^[0-9]{1,2}[ก-ฮ]{1,2}[0-9]{3,4}$"

    if re.match(pattern, text):
        return text

    return None

# Log detection
def log_detection(plate_text, filename, bbox):
    try:
        ts = datetime.now().isoformat(sep=' ', timespec='seconds')
        first = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if first:
                writer.writerow(["timestamp", "plate", "image", "bbox"])
            writer.writerow([ts, plate_text, filename, str(bbox)])
    except:
        pass

class App:
    def __init__(self, root):
        self.root = root
        root.title("Red Light Plate Tracker")
        root.geometry("1920x1080")
        self.video_src = 0
        self.cap = None
        self.running = False
        self.red_light = False
        self.line_y = 400
        self.enhance = True

        # YOLO model
        self.plate_model = None
        self.device = "cuda" if _gpu_available else "cpu"
        print(f"Device: {self.device}")
        if _yolo_available:
            # Try YOLOv11n first, fallback to YOLOv8s
            self.plate_model = ensure_yolo_model("yolov11n")
            if not self.plate_model:
                self.plate_model = ensure_yolo_model("yolov8s")

        # Auto-traffic-light detection
        self.auto_tl = False
        self.tl_interval = 2        # frames to run TL detection (cheap)
        self.tl_last_color = None

        # Tracker state
        self.trackers = {}
        self.plate_texts = {}
        self.plate_history = {}
        self.plate_id_counter = 0
        self.frame_capture_buffer = {}
        self.capture_threshold = 5
        self.detector_interval = 5
        self.frame_count = 0
        self.det_width = 960
        self.target_fps = 30
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(200, 40, False)
        self.crossed_ids = set()
        
        # Thread-safe frame queue
        self.frame_queue = Queue(maxsize=2)
        self.current_frame_tk = None

        # GUI
        ctrl = tk.Frame(root)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Button(ctrl, text="Open Video", command=self.open_file).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl, text="Camera", command=self.use_camera).pack(side=tk.LEFT, padx=2)
        self.start_btn = tk.Button(ctrl, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = tk.Button(ctrl, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.red_btn = tk.Button(ctrl, text="Red Light: OFF", bg="lightgray", command=lambda: self.toggle_red())
        self.red_btn.pack(side=tk.LEFT, padx=2)
        # Auto TL toggle
        self.auto_btn = tk.Button(ctrl, text="AutoTL: OFF", bg="lightgray", command=self.toggle_auto_tl)
        self.auto_btn.pack(side=tk.LEFT, padx=2)

        video_frame = tk.Frame(root)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Label(video_frame, bg="black", width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", lambda e: setattr(self, 'line_y', e.y))

        results_frame = tk.Frame(root, width=350, bg="white")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        tk.Label(results_frame, text="Detections", bg="white", font=("Arial", 12, "bold")).pack()
        self.results_text = tk.Text(results_frame, height=50, width=50, bg="white", fg="black")
        self.results_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        self.track_boxes = {}  # pid -> (x1,y1,x2,y2) in small-frame coords

    def open_file(self):
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4;*.avi;*.mov"), ("All", "*.*")])
        if p:
            self.video_src = p
            print(f"Selected video: {p}")
            if self.running:
                self.restart()

    def use_camera(self):
        self.video_src = 0
        print("Camera selected: index 0")
        if self.running:
            self.restart()

    def restart(self):
        # stop current loop, wait a short time, then restart with new source
        print("Restarting capture...")
        self.stop()
        # wait for loop to exit
        for _ in range(40):
            if not self.running:
                break
            time.sleep(0.05)
        # clear trackers/state
        self.trackers.clear()
        self.plate_history.clear()
        self.plate_texts.clear()
        self.frame_capture_buffer.clear()
        self.crossed_ids.clear()
        self.track_boxes.clear()
        # clear frame queue
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except Exception:
            pass
        # restart
        self.start()

    def start(self):
        if self.running:
            return
        print(f"Starting with source: {self.video_src}")
        # reset runtime counters & queue prior to capture
        self.frame_count = 0
        self.current_frame_tk = None
        # re-create queue to drop any stale frames
        self.frame_queue = Queue(maxsize=2)
        # clear state on start
        self.trackers.clear()
        self.plate_history.clear()
        self.plate_texts.clear()
        self.frame_capture_buffer.clear()
        self.crossed_ids.clear()
        self.track_boxes.clear()

        # ensure previous capture is released before opening a new source
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.cap = cv2.VideoCapture(self.video_src)
        if not self.cap.isOpened():
            print(f"ERROR: Cannot open video source: {self.video_src}")
            messagebox.showerror("Error", f"Cannot open video source: {self.video_src}")
            return
        print("Video source opened successfully")
        self.running = True
        self.crossed_ids.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        try:
            self.canvas.configure(image=None)
            self.canvas.imgtk = None
        except:
            pass
        # clear queue
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None
        print("Stopped")

    def update_canvas(self):
        """Thread-safe canvas update from main thread using queue"""
        try:
            # Get latest frame from queue (non-blocking)
            try:
                frame_tk = self.frame_queue.get_nowait()
                self.current_frame_tk = frame_tk
            except:
                pass
            
            # Update canvas if we have a frame
            if self.current_frame_tk:
                self.canvas.imgtk = self.current_frame_tk
                self.canvas.configure(image=self.current_frame_tk)
        except Exception as e:
            print(f"Canvas update error: {e}")
        
        # Schedule next update
        if self.running or self.current_frame_tk is None:
            self.root.after(30, self.update_canvas)

    def toggle_auto_tl(self):
        self.auto_tl = not self.auto_tl
        self.auto_btn.config(text=f"AutoTL: {'ON' if self.auto_tl else 'OFF'}",
                             bg=("cyan" if self.auto_tl else "lightgray"))
        if self.auto_tl:
            self.tl_last_color = None  # reset last detected color

    def set_traffic_light_color(self, color: str):
        # color in {'RED','YELLOW','GREEN'}; update UI and state
        color = (color or "").upper()
        prev = self.red_light
        self.red_light = (color == "RED")
        self.red_btn.config(text=f"Red Light: {'ON' if self.red_light else 'OFF'}",
                            bg=("red" if self.red_light else "lightgray"))
        # Notify only when actual color changed
        if color and color != self.tl_last_color:
            self.tl_last_color = color
            self.add_result(f"[TL AUTO] {color} (auto)")

    def toggle_red(self):
        # manual toggle disables auto-TL
        if self.auto_tl:
            self.auto_tl = False
            self.auto_btn.config(text="AutoTL: OFF", bg="lightgray")
        self.red_light = not self.red_light
        self.red_btn.config(text=f"Red Light: {'ON' if self.red_light else 'OFF'}", bg=("red" if self.red_light else "lightgray"))

    # Add thread-safe UI updater for results
    def add_result(self, text):
        def _insert():
            try:
                self.results_text.insert(tk.END, text + "\n")
                self.results_text.see(tk.END)
            except Exception:
                pass
        try:
            self.root.after(0, _insert)
        except Exception:
            _insert()

    def loop(self):
        print("Loop thread started")
        frame_count_local = 0
        while self.running and self.cap and self.cap.isOpened():
            start = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or read error")
                break

            orig = frame.copy()
            h, w = frame.shape[:2]
            dw = self.det_width
            dh = int(h * dw / float(w))
            small = cv2.resize(frame, (dw, dh))

            # Motion detection
            fg = self.bg_sub.apply(small)
            _, motion_mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.medianBlur(motion_mask, 5)

            # Detection every N frames
            if self.plate_model and (self.frame_count % self.detector_interval == 0):
                try:
                    detections = self.plate_model(small, conf=0.4, device=self.device, verbose=False)
                    # detections can be list of Results; each result has .boxes with .xyxy if available
                    for r in detections:
                        boxes = getattr(r, 'boxes', None)
                        if boxes is None:
                            continue
                        # boxes might be a list-like; try to iterate
                        for box in boxes:
                            # try multiple ways to get coords
                            try:
                                coords = None
                                if hasattr(box, 'xyxy'):
                                    xyxy = box.xyxy
                                    # xyxy could be tensor-like; convert
                                    x1, y1, x2, y2 = map(int, xyxy[0].tolist() if hasattr(xyxy[0], 'tolist') else xyxy[0])
                                else:
                                    # fallback: if box has .xy or .xyxy ndarray
                                    arr = getattr(box, 'xyxy', None) or getattr(box, 'xyxy', None)
                                    if arr is not None:
                                        x1, y1, x2, y2 = map(int, arr[0])
                                    else:
                                        continue
                                # now use x1,y1,x2,y2 as before
                            except Exception as e:
                                print(f"Box parse error: {e}")
                                continue
                        # Safe ROI extraction
                        y1_safe = max(0, y1)
                        y2_safe = min(motion_mask.shape[0], y2)
                        x1_safe = max(0, x1)
                        x2_safe = min(motion_mask.shape[1], x2)
                        roi = motion_mask[y1_safe:y2_safe, x1_safe:x2_safe]
                        if roi.size > 0 and float(cv2.countNonZero(roi)) / roi.size > 0.001:
                            pid = None
                            best_d = 1e9
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            for p in list(self.trackers.keys()):
                                if self.plate_history.get(p):
                                    lx, ly = self.plate_history[p][-1]
                                    d = (lx - cx)**2 + (ly - cy)**2
                                    if d < best_d:
                                        best_d, pid = d, p
                            if pid is None or best_d > 14400:
                                pid = self.plate_id_counter
                                self.plate_id_counter += 1
                                self.trackers[pid] = True
                                self.plate_history[pid] = []
                                self.frame_capture_buffer[pid] = []
                            if pid is not None:
                                # store current small-frame bbox for this tracker
                                self.track_boxes[pid] = (x1, y1, x2, y2)
                                self.plate_history[pid].append((cx, cy))
                    # --- AUTO TRAFFIC LIGHT: run every tl_interval frames ---
                    if self.auto_tl and (self.frame_count % max(1, self.tl_interval) == 0):
                        try:
                            tl_color = self.detect_traffic_light(small)
                            if tl_color:
                                self.set_traffic_light_color(tl_color)
                        except Exception as e:
                            print(f"AutoTL error: {e}")
                except Exception as e:
                    print(f"Detection error: {e}")

            # Update history and buffer frames for crossing vehicles
            for pid in list(self.trackers.keys()):
                if len(self.plate_history.get(pid, [])) >= 2:
                    # Prefer to buffer full-resolution crop using tracked small bbox -> orig mapping
                    if pid in self.track_boxes:
                        sx = float(w) / float(dw)
                        bx1, by1, bx2, by2 = self.track_boxes[pid]
                        # scale small bbox to original frame coords
                        ox1 = max(0, int(bx1 * sx))
                        oy1 = max(0, int(by1 * sx))
                        ox2 = min(w, int(bx2 * sx))
                        oy2 = min(h, int(by2 * sx))
                        # optional padding
                        pad_x = max(0, int((ox2-ox1)*0.08))
                        pad_y = max(0, int((oy2-oy1)*0.12))
                        ox1 = max(0, ox1 - pad_x); oy1 = max(0, oy1 - pad_y)
                        ox2 = min(w, ox2 + pad_x); oy2 = min(h, oy2 + pad_y)
                        # validate
                        ow = ox2 - ox1; oh = oy2 - oy1
                        if ow < 24 or oh < 12:
                            # fallback to centroid-based crop if box too small
                            hist = self.plate_history[pid]
                            cx, cy = hist[-1]
                            ox1 = max(0, int(cx * (w/dw) - 200))
                            oy1 = max(0, int(cy * (h/dh) - 100))
                            ox2 = min(w, int(cx * (w/dw) + 200))
                            oy2 = min(h, int(cy * (h/dh) + 100))
                            ow = ox2 - ox1; oh = oy2 - oy1
                        if ow > 0 and oh > 0:
                            crop = orig[oy1:oy2, ox1:ox2].copy()
                            if crop.size > 0:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                                # Only buffer if reasonably sharp to save space and avoid poles
                                if sharpness > 5.0:
                                    self.frame_capture_buffer[pid].append((crop, sharpness, (ox1, oy1, ox2, oy2)))
                                    if len(self.frame_capture_buffer[pid]) > self.capture_threshold:
                                        self.frame_capture_buffer[pid].pop(0)
                    else:
                        # fallback: centroid-based crop (legacy)
                        hist = self.plate_history[pid]
                        cx, cy = hist[-1]
                        ox1 = max(0, int(cx*(w/dw) - 200))
                        oy1 = max(0, int(cy*(h/dh) - 100))
                        ox2 = min(w, int(cx*(w/dw) + 200))
                        oy2 = min(h, int(cy*(h/dh) + 100))
                        if ox2 - ox1 > 24 and oy2 - oy1 > 12:
                            crop = orig[oy1:oy2, ox1:ox2].copy()
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                            if sharpness > 5.0:
                                self.frame_capture_buffer[pid].append((crop, sharpness, (ox1, oy1, ox2, oy2)))
                                if len(self.frame_capture_buffer[pid]) > self.capture_threshold:
                                    self.frame_capture_buffer[pid].pop(0)

        # Check crossing and report
            for pid in list(self.trackers.keys()):
                if len(self.plate_history.get(pid, [])) >= 2 and pid not in self.crossed_ids:
                    hist = self.plate_history[pid]
                    # Estimate line Y in small frame
                    canvas_h = self.canvas.winfo_height()
                    if canvas_h > 0:
                        small_line = int(self.line_y * dh / float(canvas_h))
                    else:
                        small_line = int(dh * 0.5)
                    small_line = max(0, min(small_line, dh - 1))
                    
                    if hist[-2][1] < small_line <= hist[-1][1]:
                        if self.red_light:
                            self.crossed_ids.add(pid)
                            if pid in self.frame_capture_buffer and self.frame_capture_buffer[pid]:
                                try:
                                    # choose sharpest buffered (crop, sharpness, bbox)
                                    best_crop, _, best_bbox = max(self.frame_capture_buffer[pid], key=lambda x: x[1])
                                    # verify plate region inside the crop; refine to plate area
                                    plate_region = detect_plate_region(best_crop)
                                    # If plate_region smaller than crop, we prefer plate_region for OCR and saving
                                    if plate_region.size > 0 and plate_region.shape[0] > 10 and plate_region.shape[1] > 30:
                                        final_crop = plate_region
                                    else:
                                        final_crop = best_crop
                                    enh = enhance_image(final_crop, 800)
                                    plate_text = ocr_plate(enh)
                                    ts = int(time.time())
                                    fname = os.path.join(CAPTURE_DIR, f"plate_{ts}_{pid}.jpg")
                                    cv2.imwrite(fname, final_crop)
                                    log_detection(plate_text, fname, best_bbox)
                                    msg = f"[RED LIGHT] Plate: {plate_text} | File: {os.path.basename(fname)}"
                                    self.add_result(msg)
                                    print(msg)
                                except Exception as e:
                                    print(f"Crossing capture error: {e}")

            # Draw on display
            for pid in list(self.trackers.keys()):
                if self.plate_history.get(pid):
                    pt = tuple(map(int, self.plate_history[pid][-1]))
                    cv2.circle(small, pt, 5, (0, 255, 0), -1)
                    if pid in self.plate_texts and self.plate_texts[pid]:
                        cv2.putText(small, self.plate_texts[pid], (pt[0]+10, pt[1]), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

            # Draw line
            canvas_h = self.canvas.winfo_height()
            if canvas_h > 0:
                small_line = int(self.line_y * dh / float(canvas_h))
            else:
                small_line = int(dh * 0.5)
            small_line = max(0, min(small_line, dh - 1))
            cv2.line(small, (0, small_line), (dw, small_line), (0, 0, 255) if self.red_light else (0, 255, 0), 2)

            # Prepare display image and put in queue
            display_w = 1280
            display_h = int(dh * display_w / float(dw))
            display = cv2.resize(small, (display_w, display_h))
            cv2img = cv2.cvtColor(display, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2img)
            frame_tk = ImageTk.PhotoImage(image=img)
            
            # Put frame in queue (drop old frame if queue full)
            try:
                self.frame_queue.put_nowait(frame_tk)
            except:
                pass

            elapsed = time.time() - start
            sleep_time = max(0, 1.0 / self.target_fps - elapsed)
            time.sleep(sleep_time)
            self.frame_count += 1
            frame_count_local += 1
            if frame_count_local % 30 == 0:
                print(f"Processed {frame_count_local} frames, trackers: {len(self.trackers)}")

        print("Loop thread ended")
        self.stop()

    def detect_traffic_light(self, img):
        """
        Simple HSV-based traffic-light detection on a few top ROIs.
        Returns 'RED', 'YELLOW', 'GREEN' or None.
        """
        if img is None or img.size == 0:
            return None
        h, w = img.shape[:2]
        rois = [
            img[0:int(h*0.38), int(w*0.45):int(w*0.55)],  # center upper strip
            img[0:int(h*0.32), int(w*0.08):int(w*0.22)],  # left upper cluster
            img[0:int(h*0.32), int(w*0.78):int(w*0.92)],  # right upper cluster
        ]
        h_counts = {"RED":0, "YELLOW":0, "GREEN":0}
        for roi in rois:
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # red: two ranges in HSV
            mask_r1 = cv2.inRange(hsv, (0,120,80), (8,255,255))
            mask_r2 = cv2.inRange(hsv, (170,120,80), (180,255,255))
            mask_r = cv2.bitwise_or(mask_r1, mask_r2)
            # yellow
            mask_y = cv2.inRange(hsv, (12,100,120), (35,255,255))
            # green
            mask_g = cv2.inRange(hsv, (36,80,80), (90,255,255))
            cnt_r = int(cv2.countNonZero(mask_r))
            cnt_y = int(cv2.countNonZero(mask_y))
            cnt_g = int(cv2.countNonZero(mask_g))
            # sum counts
            h_counts["RED"] += cnt_r
            h_counts["YELLOW"] += cnt_y
            h_counts["GREEN"] += cnt_g
        # choose dominant color if above threshold
        dominant = max(h_counts, key=lambda k: h_counts[k])
        total = sum(h_counts.values())
        if total <= 5:
            return None
        if h_counts[dominant] / float(total) > 0.4:
            return dominant
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.after(30, app.update_canvas)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()