# red_light_plate_tracker_fast_fixed.py
import json
import os
import cv2
import csv
from ultralytics import YOLOv10
import time
import threading
import numpy as np
from datetime import datetime
from queue import Queue, Empty
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import re
from paddleocr import PaddleOCR
import torch

# SEE NIGGA 44444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444
model = YOLOv10("yolov10n.pt")
CAPTURE_DIR = r"d:\Car plate tracking\captures"
CSV_PATH = r"d:\Car plate tracking\detections.csv"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Detect plate region
def detect_plate_region(img, min_width=50, max_width=None):
    if img is None or getattr(img, "size", 0) == 0:
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
    except Exception as e:
        print("detect_plate_region error:", e)
    return img

def enhance_plate(img):
    """Preprocess plate image for better OCR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE เพิ่ม contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    # Resize
    h, w = gray.shape
    scale = max(1, 200 // h)  # สูงสุด 200px
    gray = cv2.resize(gray, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
    
    return gray
# OCR
def ocr_plate_multi(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1: x2]
    result = ocr.ocr(frame, det=False, rec = True, cls = False)
    text = ""
    for r in result:
        #print("OCR", r)
        scores = r[0][1]
        if np.isnan(scores):
            scores = 0
        else:
            scores = int(scores * 100)
        if scores > 60:
            text = r[0][0]
    pattern = re.compile('[\W]')
    text = pattern.sub('', text)
    text = text.replace("???", "")
    text = text.replace("O", "0")
    text = text.replace("粤", "")
    return str(text)

# Clean plate text (Thai)
def clean_plate_text_thai(text):
    if not text:
        return None
    text = text.replace("I", "1").replace("|", "1").replace("O","0")
    text = text.replace(" ", "")
    pattern = r"^[0-9]{1,2}[ก-ฮ]{1,2}[0-9]{1,4}$"
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
    except Exception as e:
        print("log_detection error:", e)
def save_json(license_plates, startTime, endTime):
    #Generate individual JSON files for each 20-second interval
    interval_data = {
        "Start Time": startTime.isoformat(),
        "End Time": endTime.isoformat(),
        "License Plate": list(license_plates)
    }
    interval_file_path = "json/output_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json"
    with open(interval_file_path, 'w') as f:
        json.dump(interval_data, f, indent = 2)

    #Cummulative JSON File
    cummulative_file_path = "json/LicensePlateData.json"
    if os.path.exists(cummulative_file_path):
        with open(cummulative_file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    #Add new intervaal data to cummulative data
    existing_data.append(interval_data)

    with open(cummulative_file_path, 'w') as f:
        json.dump(existing_data, f, indent = 2)

    #Save data to SQL database
    save_to_database(license_plates, startTime, endTime)
def save_to_database(license_plates, start_time, end_time):
        conn = sqlite3.connect('licensePlatesDatabase.db')
        cursor = conn.cursor()
        for plate in license_plates:
            cursor.execute('''
                INSERT INTO LicensePlates(start_time, end_time, license_plate)
                VALUES (?, ?, ?)
            ''', (start_time.isoformat(), end_time.isoformat(), plate))
        conn.commit()
        conn.close()
    startTime = datetime.now()
    license_plates = set()
    while True:
        ret, frame = cap.read()
        if ret:
            currentTime = datetime.now()
            count += 1
            print(f"Frame Number: {count}")
            results = model.predict(frame, conf = 0.45)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    classNameInt = int(box.cls[0])
                    clsName = className[classNameInt]
                    conf = math.ceil(box.conf[0]*100)/100
                    #label = f'{clsName}:{conf}'
                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    if label:
                        license_plates.add(label)
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)
            if (currentTime - startTime).seconds >= 20:
                endTime = currentTime
                save_json(license_plates, startTime, endTime)
                startTime = currentTime
                license_plates.clear()
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('1'):
                break
        else:
            break
class App:
    def __init__(self, root):
        self.root = root
        root.title("Red Light Plate Tracker")
        root.geometry("1280x720")
        self.video_src = 0
        self.cap = None
        self.running = False
        self.red_light = False
        self.line_y = 400
        self.enhance = True

        # YOLO model + detection thread
        self.plate_model = None
        self.device = "cuda" if _gpu_available else "cpu"
        print(f"Device: {self.device}")
        if _yolo_available:
            self.plate_model = ensure_yolo_model("yolov10n")
            if not self.plate_model:
                self.plate_model = ensure_yolo_model("yolov11n")
        # detection threading
        self.det_queue = Queue(maxsize=1)
        self.latest_detections = []
        self.det_lock = threading.Lock()
        self.det_thread = None
        self.det_thread_stop = threading.Event()

        # auto TL
        self.auto_tl = False
        self.tl_interval = 2
        self.tl_last_color = None

        # trackers
        self.trackers = {}
        self.plate_texts = {}
        self.plate_history = {}
        self.plate_id_counter = 0
        self.frame_capture_buffer = {}
        self.capture_threshold = 5
        self.detector_interval = 2
        self.frame_count = 0
        self.det_width = 640
        self.target_fps = 25
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(200, 40, False)
        self.crossed_ids = set()
        self.max_track_lost = 40
        self.max_distance = 120.0

        # lane side voting: decide which half is our lane automatically
        self.lane_side = "left" # THIS IS THAILAND
        self.side_votes = {"left":0, "right":0}
        self.side_vote_threshold = 0

        # frame queue
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

    # detection worker thread
    def detection_worker(self):
        print("Detection thread started")
        while not self.det_thread_stop.is_set():
            try:
                frame_small = self.det_queue.get(timeout=0.2)
            except Empty:
                continue
            if frame_small is None:
                continue
            try:
                results = None
                try:
                    results = self.plate_model(frame_small, conf=0.35, device=self.device, verbose=False)
                except Exception as e:
                    print("YOLO inference error:", e)
                    results = None

                boxes_out = []
                if results:
                    for r in results:
                        boxes = getattr(r, "boxes", None)
                        if boxes is None:
                            continue
                        try:
                            for b in boxes:
                                if hasattr(b, "xyxy"):
                                    coords = b.xyxy[0]
                                    try:
                                        xy = coords.tolist()
                                    except:
                                        xy = [float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])]
                                    x1, y1, x2, y2 = map(int, xy)
                                else:
                                    arr = getattr(b, "xyxy", None) or getattr(b, "xyxy", None)
                                    if arr is not None:
                                        x1, y1, x2, y2 = map(int, arr[0])
                                    else:
                                        continue
                                if (x2 - x1) < 8 or (y2 - y1) < 8:
                                    continue
                                boxes_out.append((x1, y1, x2, y2))
                        except Exception as e:
                            print("Box parse exception in det worker:", e)
                            continue
                with self.det_lock:
                    self.latest_detections = boxes_out
            except Exception as e:
                print("Detection worker outer error:", e)
        print("Detection thread stopped")

    # file/camera controls
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
        print("Restarting capture...")
        self.stop()
        for _ in range(40):
            if not self.running:
                break
            time.sleep(0.02)
        self.trackers.clear()
        self.plate_history.clear()
        self.plate_texts.clear()
        self.frame_capture_buffer.clear()
        self.crossed_ids.clear()
        self.track_boxes.clear()
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except Exception:
            pass
        # reset lane votes so new video can re-decide
        self.side_votes = {"left":0, "right":0}
        self.lane_side = None
        self.start()

    def start(self):
        if self.running:
            return
        print(f"Starting with source: {self.video_src}")
        self.frame_count = 0
        self.current_frame_tk = None
        self.frame_queue = Queue(maxsize=2)
        self.trackers.clear()
        self.plate_history.clear()
        self.plate_texts.clear()
        self.frame_capture_buffer.clear()
        self.crossed_ids.clear()
        self.track_boxes.clear()

        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        # open capture
        self.cap = cv2.VideoCapture(self.video_src)
        if self.video_src == 0:
            try:
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_src, cv2.CAP_DSHOW)
            except:
                self.cap = cv2.VideoCapture(self.video_src)

        if not self.cap.isOpened():
            print(f"ERROR: Cannot open video source: {self.video_src}")
            messagebox.showerror("Error", f"Cannot open video source: {self.video_src}")
            return

        # start detection worker if model available
        if self.plate_model and (self.det_thread is None or not self.det_thread.is_alive()):
            self.det_thread_stop.clear()
            self.det_thread = threading.Thread(target=self.detection_worker, daemon=True)
            self.det_thread.start()

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        threading.Thread(target=self.loop, daemon=True).start()

    def stop(self):
        try:
            self.det_thread_stop.set()
            try:
                self.det_queue.put_nowait(None)
            except:
                pass
        except:
            pass

        try:
            self.canvas.configure(image=None)
            self.canvas.imgtk = None
        except:
            pass
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass

        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        print("Stopped")

    def update_canvas(self):
        try:
            try:
                frame_tk = self.frame_queue.get_nowait()
                self.current_frame_tk = frame_tk
            except:
                pass
            if self.current_frame_tk:
                self.canvas.imgtk = self.current_frame_tk
                self.canvas.configure(image=self.current_frame_tk)
        except Exception as e:
            print(f"Canvas update error: {e}")
        if self.running or self.current_frame_tk is None:
            self.root.after(30, self.update_canvas)

    def toggle_auto_tl(self):
        self.auto_tl = not self.auto_tl
        self.auto_btn.config(text=f"AutoTL: {'ON' if self.auto_tl else 'OFF'}",
                             bg=("cyan" if self.auto_tl else "lightgray"))
        if self.auto_tl:
            self.tl_last_color = None

    def set_traffic_light_color(self, color: str):
        color = (color or "").upper()
        self.red_light = (color == "RED")
        self.red_btn.config(text=f"Red Light: {'ON' if self.red_light else 'OFF'}",
                            bg=("red" if self.red_light else "lightgray"))
        if color and color != self.tl_last_color:
            self.tl_last_color = color
            self.add_result(f"[TL AUTO] {color} (auto)")

    def toggle_red(self):
        if self.auto_tl:
            self.auto_tl = False
            self.auto_btn.config(text="AutoTL: OFF", bg="lightgray")
        self.red_light = not self.red_light
        self.red_btn.config(text=f"Red Light: {'ON' if self.red_light else 'OFF'}", bg=("red" if self.red_light else "lightgray"))

    def add_result(self, text):
        def _insert():
            try:
                self.results_text.insert(tk.END, text + "\n")
                self.results_text.see(tk.END)
            except:
                pass
        try:
            self.root.after(0, _insert)
        except:
            _insert()

    # ---------- main loop ----------
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
            # small_h, small_w for mapping
            small_h, small_w = small.shape[:2]

            # Motion detection
            fg = self.bg_sub.apply(small)
            _, motion_mask = cv2.threshold(fg, 180, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.medianBlur(motion_mask, 3)

            # Submit small frame for background detection every detector_interval frames
            if self.plate_model and (self.frame_count % max(1, self.detector_interval) == 0):
                try:
                    if not self.det_queue.empty():
                        try:
                            _ = self.det_queue.get_nowait()
                        except:
                            pass
                    self.det_queue.put_nowait(small.copy())
                except:
                    pass

            # Read latest detections from det thread
            boxes = []
            with self.det_lock:
                if self.latest_detections:
                    boxes = list(self.latest_detections)

            # Fallback proposals
            if not boxes:
                gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_s, 80, 200)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    if cw < 20 or ch < 8:
                        continue
                    aspect = cw / max(1, ch)
                    if 1.5 < aspect < 10.0:
                        x1 = max(0, x - 3); y1 = max(0, y - 3)
                        x2 = min(small_w, x + cw + 3); y2 = min(small_h, y + ch + 3)
                        boxes.append((x1, y1, x2, y2))

            # Filter boxes by motion_mask
            filtered_boxes = []
            for (x1, y1, x2, y2) in boxes:
                y1s = max(0, int(y1)); y2s = min(motion_mask.shape[0], int(y2))
                x1s = max(0, int(x1)); x2s = min(motion_mask.shape[1], int(x2))
                if y2s <= y1s or x2s <= x1s:
                    continue
                roi = motion_mask[y1s:y2s, x1s:x2s]
                if roi.size > 0 and float(cv2.countNonZero(roi)) / roi.size > 0.0005:
                    filtered_boxes.append((x1, y1, x2, y2))
            boxes = filtered_boxes


            # Filter boxes by lane_side (if decided) to ignore opposite lane
            kept = []
            for (x1, y1, x2, y2) in boxes:
                cx = (x1 + x2) / 2.0
                if cx < (small_w / 2.0):  # ซ้าย
                    kept.append((x1, y1, x2, y2))
            boxes = kept

            # ---------- Tracker matching (centroid tracker) ----------
            detections_centroids = []
            for (x1, y1, x2, y2) in boxes:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections_centroids.append(((x1, y1, x2, y2), (cx, cy)))

            unmatched_dets = set(range(len(detections_centroids)))
            matched_tracker_ids = set()
            tracker_items = list(self.trackers.items())
            for pid, tinfo in tracker_items:
                tpos = tinfo.get("pos")
                if tpos is None:
                    tinfo["age"] = tinfo.get("age", 0) + 1
                    continue
                best_det = None
                best_d = float("inf")
                for di in list(unmatched_dets):
                    _, (cx, cy) = detections_centroids[di]
                    d = (tpos[0] - cx)**2 + (tpos[1] - cy)**2
                    if d < best_d:
                        best_d = d
                        best_det = di
                if best_det is not None and best_d < (self.max_distance ** 2):
                    box, centroid = detections_centroids[best_det]
                    unmatched_dets.remove(best_det)
                    self.track_boxes[pid] = tuple(map(int, box))
                    self.plate_history.setdefault(pid, []).append(centroid)
                    self.trackers[pid]["pos"] = centroid
                    self.trackers[pid]["last_seen"] = self.frame_count
                    self.trackers[pid]["age"] = 0
                    matched_tracker_ids.add(pid)
                else:
                    self.trackers[pid]["age"] = self.trackers[pid].get("age", 0) + 1

            for di in list(unmatched_dets):
                box, centroid = detections_centroids[di]
                pid = self.plate_id_counter
                self.plate_id_counter += 1
                self.track_boxes[pid] = tuple(map(int, box))
                self.trackers[pid] = {"pos": centroid, "last_seen": self.frame_count, "age": 0}
                self.plate_history[pid] = [centroid]
                self.frame_capture_buffer[pid] = []
                matched_tracker_ids.add(pid)

            # cleanup old trackers (also remove frame_capture_buffer to avoid memory leak)
            for pid in list(self.trackers.keys()):
                if self.trackers[pid].get("age", 0) > self.max_track_lost:
                    try:
                        del self.trackers[pid]
                    except:
                        pass
                    try:
                        del self.plate_history[pid]
                    except:
                        pass
                    try:
                        del self.track_boxes[pid]
                    except:
                        pass
                    try:
                        del self.frame_capture_buffer[pid]
                    except:
                        pass
                    try:
                        if pid in self.plate_texts:
                            del self.plate_texts[pid]
                    except:
                        pass

            # Update history and capture buffer (map small->orig using sx,sy)
            sx = float(w) / float(small_w)
            sy = float(h) / float(small_h)
            for pid in list(self.trackers.keys()):
                if len(self.plate_history.get(pid, [])) >= 1:
                    if pid in self.track_boxes:
                        bx1, by1, bx2, by2 = self.track_boxes[pid]
                        ox1 = max(0, int(bx1 * sx))
                        oy1 = max(0, int(by1 * sy))
                        ox2 = min(w, int(bx2 * sx))
                        oy2 = min(h, int(by2 * sy))
                        pad_x = max(0, int((ox2-ox1)*0.08))
                        pad_y = max(0, int((oy2-oy1)*0.12))
                        ox1 = max(0, ox1 - pad_x); oy1 = max(0, oy1 - pad_y)
                        ox2 = min(w, ox2 + pad_x); oy2 = min(h, oy2 + pad_y)
                        ow = ox2 - ox1; oh = oy2 - oy1
                        if ow < 24 or oh < 12:
                            hist = self.plate_history[pid]
                            cx, cy = hist[-1]
                            ox1 = max(0, int(cx * (w/small_w) - 200))
                            oy1 = max(0, int(cy * (h/small_h) - 100))
                            ox2 = min(w, int(cx * (w/small_w) + 200))
                            oy2 = min(h, int(cy * (h/small_h) + 100))
                            ow = ox2 - ox1; oh = oy2 - oy1
                        if ow > 0 and oh > 0:
                            crop = orig[oy1:oy2, ox1:ox2].copy()
                            if crop.size > 0:
                                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                                if sharpness > 12.0:   # increased threshold for sharper crops
                                    self.frame_capture_buffer[pid].append((crop, sharpness, (ox1, oy1, ox2, oy2)))
                                    if len(self.frame_capture_buffer[pid]) > self.capture_threshold:
                                        self.frame_capture_buffer[pid].pop(0)

            # Check crossing and report
            for pid in list(self.trackers.keys()):
                if len(self.plate_history.get(pid, [])) >= 2 and pid not in self.crossed_ids:
                    hist = self.plate_history[pid]
                    canvas_h = self.canvas.winfo_height()
                    if canvas_h > 0:
                        small_line = int(self.line_y * small_h / float(canvas_h))
                    else:
                        small_line = int(small_h * 0.5)
                    small_line = max(0, min(small_line, small_h - 1))

                    if hist[-2][1] < small_line <= hist[-1][1]:
                        if self.red_light:
                            self.crossed_ids.add(pid)
                            # capture best crop
                            if pid in self.frame_capture_buffer and self.frame_capture_buffer[pid]:
                                try:
                                    best_crop, _, best_bbox = max(self.frame_capture_buffer[pid], key=lambda x: x[1])
                                    plate_region = detect_plate_region(best_crop)
                                    if plate_region is None or plate_region.size == 0 or plate_region.shape[0] <= 10 or plate_region.shape[1] <= 30:
                                        final_crop = best_crop
                                    else:
                                        final_crop = plate_region
                                    enh = enhance_image(final_crop, 800)
                                    plate_text = ocr_plate_multi(enh)
                                    cleaned = clean_plate_text_thai(plate_text)
                                    ts = int(time.time())
                                    fname = os.path.join(CAPTURE_DIR, f"plate_{ts}_{pid}.jpg")
                                    cv2.imwrite(fname, final_crop)
                                    log_detection(cleaned if cleaned else plate_text, fname, best_bbox)
                                    msg = f"[RED LIGHT] Plate: {cleaned if cleaned else plate_text} | File: {os.path.basename(fname)}"
                                    self.add_result(msg)
                                    print(msg)
                                except Exception as e:
                                    print(f"Crossing capture error: {e}")

            # Draw trackers and texts
            for pid in list(self.trackers.keys()):
                if self.plate_history.get(pid):
                    pt = tuple(map(int, self.plate_history[pid][-1]))
                    cv2.circle(small, pt, 4, (0, 255, 0), -1)
                    if pid in self.plate_texts and self.plate_texts[pid]:
                        cv2.putText(small, self.plate_texts[pid], (pt[0]+10, pt[1]),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)

            # Draw detection boxes (debug/visual)
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(small, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

            # Draw line
            canvas_h = self.canvas.winfo_height()
            if canvas_h > 0:
                small_line = int(self.line_y * small_h / float(canvas_h))
            else:
                small_line = int(small_h * 0.5)
            small_line = max(0, min(small_line, small_h - 1))
            cv2.line(small, (0, small_line), (small_w, small_line), (0, 0, 255) if self.red_light else (0, 255, 0), 2)

            # Prepare display image and put in queue (clear old to avoid backlog)
            display_w = 960
            display_h = int(small_h * display_w / float(small_w))
            display = cv2.resize(small, (display_w, display_h))
            cv2img = cv2.cvtColor(display, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2img)
            frame_tk = ImageTk.PhotoImage(image=img)
            try:
                # clear queue quickly and push current frame to keep UI responsive
                with self.frame_queue.mutex:
                    self.frame_queue.queue.clear()
                self.frame_queue.put_nowait(frame_tk)
            except:
                pass

            elapsed = time.time() - start
            sleep_time = max(0, 1.0 / self.target_fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            self.frame_count += 1
            frame_count_local += 1
            if frame_count_local % 30 == 0:
                print(f"Processed {frame_count_local} frames, trackers: {len(self.trackers)}, lane_side={self.lane_side}")

        print("Loop thread ended")
        self.stop()

    def detect_traffic_light(self, img):
        if img is None or getattr(img, "size", 0) == 0:
            return None
        h, w = img.shape[:2]
        rois = [
            img[0:int(h*0.38), int(w*0.45):int(w*0.55)],
            img[0:int(h*0.32), int(w*0.08):int(w*0.22)],
            img[0:int(h*0.32), int(w*0.78):int(w*0.92)],
        ]
        h_counts = {"RED":0, "YELLOW":0, "GREEN":0}
        for roi in rois:
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_r1 = cv2.inRange(hsv, (0,120,80), (8,255,255))
            mask_r2 = cv2.inRange(hsv, (170,120,80), (180,255,255))
            mask_r = cv2.bitwise_or(mask_r1, mask_r2)
            mask_y = cv2.inRange(hsv, (12,100,120), (35,255,255))
            mask_g = cv2.inRange(hsv, (36,80,80), (90,255,255))
            cnt_r = int(cv2.countNonZero(mask_r))
            cnt_y = int(cv2.countNonZero(mask_y))
            cnt_g = int(cv2.countNonZero(mask_g))
            h_counts["RED"] += cnt_r
            h_counts["YELLOW"] += cnt_y
            h_counts["GREEN"] += cnt_g
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

