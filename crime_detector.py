"""
Crime Detection Core Engine
----------------------------
- Person/Object detection: YOLOv8 (ultralytics)
- Face Recognition: OpenCV LBPH recognizer (cv2.face) matched against wanted_faces/
  (No dlib/face_recognition - avoids needing Visual Studio C++ build tools on Windows)
- Fight detection: heuristic based on person-bbox overlap + rapid movement
- Alerts logged to alerts_log.csv

Folder structure expected:
crime_system/
  wanted_faces/        <- put photos here, see _load_wanted_faces() for naming rules
  crime_detector.py
  streamlit_app.py
  alerts_log.csv        <- auto created

wanted_faces/ naming rules (either works):
  Option A (simple):  wanted_faces/rahul.jpg, wanted_faces/suresh.jpg   (1 photo per person)
  Option B (better accuracy): wanted_faces/rahul/1.jpg, wanted_faces/rahul/2.jpg ...
                               wanted_faces/suresh/1.jpg ...  (multiple photos per person, per-person subfolder)

CAMERA FIX (Windows):
  The default MSMF backend often opens the webcam but cannot grab frames.
  Use open_camera() from this file everywhere (also in streamlit_app.py):
      from crime_detector import CrimeDetector, open_camera
      cap = open_camera(0)                       # webcam
      cap = open_camera("rtsp://user:pass@ip/…") # RTSP stream
"""

import os

# Must be set BEFORE cv2 is imported: disable the flaky MSMF backend on Windows.
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import re
import csv
import time
import warnings
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import urllib.request
import easyocr
from ultralytics import YOLO
from crime_database import AlertStore

# Hide the noisy (harmless) PyTorch quantization deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"


def open_camera(source=0, width=640, height=480):
    """Open a webcam (index) or a stream (RTSP/HTTP URL / video file) reliably.

    - Webcam on Windows -> uses DirectShow (CAP_DSHOW). This fixes the
      'MSMF: can't grab frame' problem.
    - RTSP / URL / file  -> normal VideoCapture (DSHOW must NOT be used for these).
    Returns an opened cv2.VideoCapture, or None if it could not be opened.
    """
    is_stream = isinstance(source, str) and not source.strip().isdigit()

    if is_stream:
        cap = cv2.VideoCapture(source)
    else:
        index = int(source)
        cap = None
        if os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(index)

    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not is_stream:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def resize_keep_aspect(frame, max_width=640):
    """Shrink big frames so YOLO/OCR/face-detection run faster on CPU."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)))


def get_cascade_path():
    """Find the face-detection cascade file, or download it locally if the opencv
    package didn't ship it (happens with some opencv-contrib-python builds on Windows)."""
    bundled = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(bundled):
        return bundled

    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    if not os.path.exists(local_path):
        print("[CrimeDetector] Cascade file not found in opencv package, downloading it once...")
        urllib.request.urlretrieve(CASCADE_URL, local_path)
        print(f"[CrimeDetector] Saved cascade to {local_path}")
    return local_path


class CrimeDetector:
    def __init__(self, wanted_faces_dir="wanted_faces", log_file="alerts_log.csv",
                 fight_distance_thresh=100, fight_motion_thresh=15, fight_history=10,
                 face_confidence_thresh=55, yolo_confidence=0.45,
                 yolo_iou=0.5, alert_cooldown=10, crowd_limit=8,
                 location="Local Camera", source="webcam", database_path="crime_records.db"):
        # YOLOv8 nano - fast enough for CPU demo. Detects: person, knife*, backpack, car etc (COCO classes)
        self.yolo = YOLO("yolov8n.pt")

        # Haar cascade for face detection (ships with opencv, with an auto-download fallback)
        cascade_path = get_cascade_path()
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load face cascade file: {cascade_path}")

        # LBPH recognizer for matching against wanted list
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_to_name = {}
        self.trained = False
        self._train_wanted_faces(wanted_faces_dir)

        # lower distance = more confident match. LBPH "confidence" is actually a distance
        # (0 = perfect match); typical usable threshold is 60-90. Tune based on your demo lighting.
        self.face_confidence_thresh = face_confidence_thresh
        self.yolo_confidence = yolo_confidence
        self.yolo_iou = yolo_iou
        self.alert_cooldown = alert_cooldown
        self.last_alert = {}
        self.crowd_limit = crowd_limit
        self.location = location
        self.source = source

        # Fight detection heuristic params
        self.fight_distance_thresh = fight_distance_thresh   # px between two person centers
        self.fight_motion_thresh = fight_motion_thresh       # px movement per frame considered "fast"
        self.prev_centers = {}   # id -> deque of last N centers
        self.fight_history = fight_history
        self.fight_streak = 0
        self.fight_required_frames = 2
        self.previous_gray = None

        # Alert log
        self.log_file = log_file
        self.alert_store = AlertStore(log_file, database_path, location, source)
        self.current_stats = {"person_count": 0, "crowd_alert": False, "vehicle_count": 0, "weapon_count": 0}

        # left-object detection: track stationary bags
        self.bag_first_seen = {}   # bbox_key -> first_seen_time
        self.left_object_seconds = 15  # alert if bag stationary this long

        # ANPR (number plate) setup
        self.ocr_reader = easyocr.Reader(["en"], gpu=False)
        self.stolen_plates = self._load_stolen_plates("stolen_plates.txt")
        self.last_plate_read = {}   # track_id -> last time we ran OCR on it
        self.plate_ocr_interval = 2.0  # seconds between OCR attempts per vehicle (OCR is slow)
        self.known_plates_seen = {}  # track_id -> last read plate text (for on-screen label)

    def _load_stolen_plates(self, path):
        plates = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        plates.add(re.sub(r"[\s-]", "", line).upper())
        return plates

    def _clean_plate_text(self, text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _resolve_known_plate(self, candidate):
        """Resolve a partial OCR result against the configured stolen-plate list."""
        candidate = self._clean_plate_text(candidate)
        if candidate in self.stolen_plates:
            return candidate
        if len(candidate) < 4:
            return candidate
        for known in self.stolen_plates:
            if known.endswith(candidate) or known.startswith(candidate):
                return known
        return candidate

    def read_plate(self, frame, x1, y1, x2, y2, fast=True):
        """Run OCR on a vehicle crop, return cleaned plate text or None."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants = [crop]
        if not fast:
            variants.extend([
                gray,
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 11),
            ])
        results = []
        for variant in variants:
            enlarged = cv2.resize(variant, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            try:
                results.extend(self.ocr_reader.readtext(
                    enlarged, detail=1, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
            except Exception:
                continue

        if not results:
            return None

        print(f"[ANPR debug] raw OCR fragments: {[(t, round(float(c),2)) for _, t, c in results]}")

        if fast:
            for _, text, confidence in sorted(results, key=lambda item: item[2], reverse=True):
                cleaned = self._clean_plate_text(text)
                resolved = self._resolve_known_plate(cleaned)
                if resolved != cleaned:
                    return resolved

        # Words that commonly appear on/near plates but aren't part of the plate number itself
        noise_words = {"IND", "INDIA", "HERO", "ARMY", "BOXSTAR"}

        # Indian plates often get OCR'd as multiple fragments (e.g. "MH", "11", "CE", "3302"),
        # sometimes across two lines (e.g. "MH 11 CE" on top, "3302" below).
        # Sort by each fragment's box center (top-to-bottom, then left-to-right) for correct reading order.
        confident = [r for r in results if r[2] > 0.4 and self._clean_plate_text(r[1]) not in noise_words]

        def box_center(box):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            return (sum(ys) / len(ys), sum(xs) / len(xs))

        confident.sort(key=lambda r: box_center(r[0]))
        combined = "".join(self._clean_plate_text(text) for _, text, conf in confident)

        if 6 <= len(combined) <= 13:
            return self._resolve_known_plate(combined)

        # Fallback 1: longest single fragment, in case combining added noise
        best_text, best_len = None, 0
        for _, text, conf in results:
            cleaned = self._clean_plate_text(text)
            if 6 <= len(cleaned) <= 12 and conf > 0.4 and len(cleaned) > best_len:
                best_text, best_len = cleaned, len(cleaned)
        if best_text:
            return self._resolve_known_plate(best_text)

        # Fallback 2: any fragment that already looks like a plate (letters+digits mixed),
        # even at lower confidence - handles cases like "MH11CN8950" read as one blurry chunk.
        def is_plate_like(cleaned):
            has_letter = any(c.isalpha() for c in cleaned)
            has_digit = any(c.isdigit() for c in cleaned)
            return has_letter and has_digit and 5 <= len(cleaned) <= 14

        plate_like = [r for r in results if self._clean_plate_text(r[1]) not in noise_words
                      and is_plate_like(self._clean_plate_text(r[1]))]
        if plate_like:
            best = max(plate_like, key=lambda r: r[2])
            return self._resolve_known_plate(best[1])

        return None

    # ---------- Setup ----------
    def _train_wanted_faces(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            return

        faces, labels = [], []
        label_id = 0

        entries = sorted(os.listdir(folder))
        for entry in entries:
            entry_path = os.path.join(folder, entry)

            if os.path.isdir(entry_path):
                # Option B: subfolder per person with multiple photos
                name = entry
                imgs = [f for f in os.listdir(entry_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if not imgs:
                    continue
                self.label_to_name[label_id] = name
                for img_name in imgs:
                    face_img = self._read_face_gray(os.path.join(entry_path, img_name))
                    if face_img is not None:
                        faces.append(face_img)
                        labels.append(label_id)
                label_id += 1

            elif entry.lower().endswith((".jpg", ".jpeg", ".png")):
                # Option A: single photo per person, filename = name
                name = os.path.splitext(entry)[0]
                face_img = self._read_face_gray(entry_path)
                if face_img is not None:
                    self.label_to_name[label_id] = name
                    faces.append(face_img)
                    labels.append(label_id)
                    label_id += 1

        if faces:
            self.recognizer.train(faces, np.array(labels))
            self.trained = True
            print(f"[CrimeDetector] Trained on {len(faces)} face image(s) for {len(self.label_to_name)} wanted person(s).")
        else:
            print("[CrimeDetector] No wanted-face photos found yet - add images to wanted_faces/.")

    def _read_face_gray(self, path, size=(200, 200)):
        """Load an image, detect the first face, return cropped+resized grayscale face for training."""
        img = cv2.imread(path)
        if img is None:
            return None
        # Big phone photos: shrink so face detection is stable and fast
        longest = max(img.shape[:2])
        if longest > 1000:
            scale = 1000.0 / longest
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        if len(faces) == 0:
            print(f"[CrimeDetector] No face found in training photo: {path}")
            return None
        # use the LARGEST face (not just the first one found)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, size)
        return cv2.equalizeHist(face)

    def _log_alert(self, alert_type, details):
        self.alert_store.log(alert_type, details)

    def _log_alert_once(self, alert_type, details):
        key = f"{alert_type}:{details.split(' (confidence:', 1)[0]}"
        now = time.time()
        if now - self.last_alert.get(key, 0) >= self.alert_cooldown:
            self.last_alert[key] = now
            self._log_alert(alert_type, details)

    # ---------- Face recognition ----------
    def _detect_faces(self, gray):
        """Haar face detection with a relaxed second pass if the strict pass finds nothing."""
        eq = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            eq, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                eq, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
        return faces

    def scan_faces(self, frame):
        """Detect every face in the frame and compare with the wanted list.
        Returns list of dicts: {box, name (closest wanted person), distance, matched}.
        Lower distance = better match. matched=True when distance < face_confidence_thresh."""
        results = []
        if not self.trained:
            return results

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in self._detect_faces(gray):
            x, y, w, h = int(x), int(y), int(w), int(h)
            face_crop = cv2.equalizeHist(cv2.resize(gray[y:y + h, x:x + w], (200, 200)))
            label_id, distance = self.recognizer.predict(face_crop)
            name = self.label_to_name.get(label_id, "unknown")
            matched = distance < self.face_confidence_thresh and label_id in self.label_to_name
            results.append({"box": (x, y, x + w, y + h), "name": name,
                            "distance": float(distance), "matched": matched})
            if matched:
                self._log_alert_once("WANTED_CRIMINAL", f"Name: {name} (confidence: {distance:.1f})")
        return results

    def check_wanted_faces(self, frame):
        """Returns list of (name, box) for matched wanted faces in this frame."""
        return [(f["name"], f["box"]) for f in self.scan_faces(frame) if f["matched"]]

    # ---------- Object/person detection + fight + left object ----------
    def process_frame(self, frame):
        results = self.yolo.track(
            frame, persist=True, verbose=False,
            conf=self.yolo_confidence, iou=self.yolo_iou,
            classes=[0, 2, 3, 5, 7, 24, 26, 28, 39, 43])
        # class ids (COCO): 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck,
        #                    24=backpack, 26=handbag, 28=suitcase, 39=bottle, 43=knife

        person_boxes = {}   # track_id -> (center, bounding-box)
        annotated = frame.copy()
        plate_alerts = []   # list of (plate_text, is_stolen)
        vehicle_count = 0
        weapon_count = 0

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                track_id = int(boxes.id[i]) if boxes.id is not None else i
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if cls_id == 0:  # person
                    person_boxes[track_id] = ((cx, cy), (x1, y1, x2, y2))
                    color = (0, 255, 0)
                    label = f"Person {track_id}"
                elif cls_id in (2, 3, 5, 7):  # vehicle -> ANPR
                    vehicle_count += 1
                    color = (255, 255, 0)
                    vehicle_type = self.yolo.names[cls_id]
                    label = vehicle_type

                    now = time.time()
                    should_read = now - self.last_plate_read.get(track_id, 0) > self.plate_ocr_interval
                    if should_read:
                        self.last_plate_read[track_id] = now
                        # Expand the box downward/sideways a bit - YOLO's vehicle box sometimes
                        # cuts off right before the plate on close-up/unusual-angle shots.
                        box_h = y2 - y1
                        box_w = x2 - x1
                        exp_x1 = x1 - box_w * 0.1
                        exp_x2 = x2 + box_w * 0.1
                        exp_y2 = y2 + box_h * 0.6
                        plate_text = self.read_plate(frame, exp_x1, y1, exp_x2, exp_y2, fast=True)
                        if plate_text:
                            self.known_plates_seen[track_id] = plate_text
                            is_stolen = plate_text in self.stolen_plates
                            plate_alerts.append((plate_text, is_stolen))
                            if is_stolen:
                                self._log_alert_once("STOLEN_VEHICLE", f"Plate: {plate_text}")
                            else:
                                self._log_alert_once("PLATE_READ", f"Plate: {plate_text}")

                    seen_plate = self.known_plates_seen.get(track_id)
                    if seen_plate:
                        is_stolen = seen_plate in self.stolen_plates
                        color = (0, 0, 255) if is_stolen else (255, 255, 0)
                        label = f"{vehicle_type}: {seen_plate}" + (" STOLEN!" if is_stolen else "")
                elif cls_id == 43:  # knife
                    weapon_count += 1
                    color = (0, 0, 255)
                    label = "WEAPON!"
                    self._log_alert_once("WEAPON_DETECTED", f"box:{int(x1)},{int(y1)},{int(x2)},{int(y2)}")
                elif cls_id in (24, 26, 28):  # bag/backpack/suitcase -> left object check
                    color = (255, 165, 0)
                    label = "Bag"
                    key = f"{int(x1/20)}_{int(y1/20)}"  # coarse spatial bucket
                    now = time.time()
                    if key not in self.bag_first_seen:
                        self.bag_first_seen[key] = now
                    elif now - self.bag_first_seen[key] > self.left_object_seconds:
                        label = "LEFT OBJECT ALERT!"
                        self._log_alert_once("LEFT_OBJECT", f"box:{int(x1)},{int(y1)},{int(x2)},{int(y2)}")
                else:
                    color = (200, 200, 0)
                    label = self.yolo.names[cls_id]

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_motion = None
        if self.previous_gray is not None and self.previous_gray.shape == gray_frame.shape:
            frame_motion = cv2.absdiff(self.previous_gray, gray_frame)
        self.previous_gray = gray_frame

        fight_detected = self._check_fight(person_boxes, frame_motion)
        crowd_alert = len(person_boxes) >= self.crowd_limit
        self.current_stats = {
            "person_count": len(person_boxes),
            "crowd_alert": crowd_alert,
            "vehicle_count": vehicle_count,
            "weapon_count": weapon_count,
        }
        if crowd_alert:
            cv2.putText(annotated, f"CROWD ALERT: {len(person_boxes)} people", (30, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            self._log_alert_once("CROWD_ALERT", f"{len(person_boxes)} people at {self.location}")
        if fight_detected:
            cv2.putText(annotated, "FIGHT DETECTED!", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            self._log_alert_once("FIGHT_DETECTED", f"{len(person_boxes)} persons close & moving fast")

        return annotated, fight_detected, plate_alerts

    def _check_fight(self, person_boxes, frame_motion=None):
        """Very simple heuristic: two people close together AND moving fast = possible fight.
        For a real dissertation-grade model, replace this with a trained CNN+LSTM
        (train on UCF-Crime / CCTV-Fights dataset) - this heuristic is a good working demo baseline."""
        ids = list(person_boxes.keys())
        fast_moving = set()

        for pid, person_data in person_boxes.items():
            center = person_data[0]
            self.prev_centers.setdefault(pid, deque(maxlen=self.fight_history))
            hist = self.prev_centers[pid]
            if hist:
                dx = center[0] - hist[-1][0]
                dy = center[1] - hist[-1][1]
                dist_moved = (dx ** 2 + dy ** 2) ** 0.5
                if dist_moved > self.fight_motion_thresh:
                    fast_moving.add(pid)
            if frame_motion is not None:
                x1, y1, x2, y2 = [max(0, int(value)) for value in person_data[1]]
                y2 = min(frame_motion.shape[0], y2)
                x2 = min(frame_motion.shape[1], x2)
                if x2 > x1 and y2 > y1:
                    movement_ratio = (frame_motion[y1:y2, x1:x2] > 25).mean()
                    if movement_ratio > 0.015:
                        fast_moving.add(pid)
            hist.append(center)

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                p1, p2 = person_boxes[ids[i]], person_boxes[ids[j]]
                center1, box1 = p1
                center2, box2 = p2
                dist = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
                height1 = max(1, box1[3] - box1[1])
                height2 = max(1, box2[3] - box2[1])
                scaled_distance = max(self.fight_distance_thresh, 1.2 * ((height1 + height2) / 2))
                horizontal_gap = max(box1[0] - box2[2], box2[0] - box1[2], 0)
                vertical_gap = max(box1[1] - box2[3], box2[1] - box1[3], 0)
                boxes_interacting = horizontal_gap <= 0.35 * max(height1, height2) and vertical_gap <= 0.5 * max(height1, height2)
                close_enough = dist < scaled_distance or boxes_interacting
                if close_enough and (ids[i] in fast_moving or ids[j] in fast_moving):
                    self.fight_streak += 1
                    return self.fight_streak >= self.fight_required_frames

        self.fight_streak = 0
        return False


if __name__ == "__main__":
    # Quick local test using webcam (run this file directly to sanity-check before wiring into Streamlit)
    PROCESS_EVERY_N = 2      # run YOLO on every 2nd frame (faster on CPU)
    FACE_EVERY_N = 4         # run face recognition on every 4th frame
    MAX_FAILS = 30           # reopen camera after this many failed reads

    detector = CrimeDetector()
    cap = open_camera(0)
    if cap is None:
        raise SystemExit("Camera open nahi hua. Dusra app band karo / camera index badlo / "
                         "Windows Privacy settings madhe camera access ON kara.")

    frame_no = 0
    fails = 0
    last_annotated = None
    last_faces = []

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fails += 1
            if fails > MAX_FAILS:
                print("[Camera] Too many failed reads, reopening camera...")
                cap.release()
                cap = open_camera(0)
                if cap is None:
                    print("[Camera] Could not reopen camera.")
                    break
                fails = 0
            time.sleep(0.05)
            continue
        fails = 0
        frame_no += 1

        frame = resize_keep_aspect(frame, 640)

        if frame_no % PROCESS_EVERY_N == 0 or last_annotated is None:
            last_annotated, _, _ = detector.process_frame(frame)
        if frame_no % FACE_EVERY_N == 0:
            last_faces = detector.check_wanted_faces(frame)

        display = last_annotated.copy() if last_annotated is not None else frame
        for name, (l, t, r, b) in last_faces:
            cv2.rectangle(display, (l, t), (r, b), (0, 0, 255), 2)
            cv2.putText(display, f"WANTED: {name}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Crime System AI", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()