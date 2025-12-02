from flask import Flask, Response, render_template_string, jsonify
import threading, queue, time, math, collections
import cv2, numpy as np
from openvino.runtime import Core
from pymongo import MongoClient
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import colorsys

# Load environment variables from .env file
load_dotenv()

# ------------------------------------------
# CONFIG
# ------------------------------------------
CAM_SOURCES = ["videos/video7.mp4"]  
DETECTION_MODEL = "models/person-detection-retail-0013.xml"
REID_MODEL = "models/person-reidentification-retail-0287.xml"

REID_SIM_THRESHOLD = 0.62
REID_UPDATE_MOMENTUM = 0.6
FRAME_THROTTLE = 0.03

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "product_analytics")

ZONES_COLLECTION = os.getenv("ZONES_COLLECTION", "zones")
FOOTFALL_COL = os.getenv("FOOTFALL_COL", "product_footfall")
DWELL_COL = os.getenv("DWELL_COL", "product_dwell")
HOURLY_COL = os.getenv("HOURLY_COL", "hourly_stats")
DAILY_COL = os.getenv("DAILY_COL", "daily_stats")

HEATMAP_DECAY = 0.97
HEATMAP_KERNEL_SIZE = 121
PATH_MAX_LEN = 40

ZONE_THRESHOLDS = [
    (600, (0, 0, 255)),    # >=10 minutes -> red
    (300, (0, 165, 255)),  # >=5 minutes  -> orange
    (120, (0, 255, 255)),  # >=2 minutes  -> yellow
]

BOX_COLOR = (0, 255, 0)
DEFAULT_ZONE_DOT_COLOR = (0, 200, 0)

# ------------------------------------------
# FLASK + DB INIT
# ------------------------------------------
app = Flask(__name__)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

zones_col = db[ZONES_COLLECTION]
footfall_col = db[FOOTFALL_COL]
dwell_col = db[DWELL_COL]
hourly_col = db[HOURLY_COL]
daily_col = db[DAILY_COL]

# ------------------------------------------
# OPENVINO MODELS
# ------------------------------------------
core = Core()

det_model = core.read_model(DETECTION_MODEL)
compiled_det = core.compile_model(det_model, "CPU")
det_out = compiled_det.output(0)

reid_model = core.read_model(REID_MODEL)
compiled_reid = core.compile_model(reid_model, "CPU")
reid_out = compiled_reid.output(0)

# ------------------------------------------
# TRACKING STATE
# ------------------------------------------
cam_queues = []
cam_threads = []
stop_event = threading.Event()

# Global person registry for multi-camera ReID
registry = []
registry_lock = threading.Lock()
next_global_id = 0

# Product analytics
product_stats = {}
product_stats_lock = threading.Lock()

# Path tracing for each person
paths = collections.defaultdict(lambda: collections.deque(maxlen=PATH_MAX_LEN))
paths_lock = threading.Lock()

# Per-person visualization and zone state
person_color_cache = {}
person_zone_state = {}
person_zone_lock = threading.Lock()

# Persistent zone dots for the whole day (product_id -> list of (timestamp, color))
zone_dot_history = collections.defaultdict(list)
zone_dot_history_lock = threading.Lock()
current_day = datetime.now().date()

# Heatmap accumulators per camera
heatmap_accumulators = {}
heatmap_lock = threading.Lock()

# ------------------------------------------
# UTILS
# ------------------------------------------
def l2_norm(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def hsv_to_bgr(h, s, v):
    """Convert HSV floats (0-1) to BGR tuple"""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))

def get_unique_person_color(pid):
    """Return deterministic unique color for a person id"""
    if pid in person_color_cache:
        return person_color_cache[pid]

    # Use golden ratio to distribute hues
    golden_ratio = 0.6180339887498949
    hue = (pid * golden_ratio) % 1.0
    color = hsv_to_bgr(hue, 0.75, 1.0)
    person_color_cache[pid] = color
    return color

def update_person_zone_state(pid, zone_ids, timestamp):
    """Track which zone a person is in and when they entered"""
    active_zone = zone_ids[0] if zone_ids else None
    with person_zone_lock:
        state = person_zone_state.get(pid)
        if not state:
            person_zone_state[pid] = {
                "zone": active_zone,
                "start": timestamp if active_zone else None,
                "last_seen": timestamp if active_zone else None
            }
            return

        if active_zone is None:
            state["zone"] = None
            state["start"] = None
            state["last_seen"] = None
            return

        if state["zone"] != active_zone or state["start"] is None:
            state["zone"] = active_zone
            state["start"] = timestamp

        state["last_seen"] = timestamp

def get_zone_dwell_duration(pid):
    """Return how long (seconds) a person has been in their current zone"""
    with person_zone_lock:
        state = person_zone_state.get(pid)
        if not state or not state.get("zone") or not state.get("start"):
            return 0.0
        ref_time = state.get("last_seen") or time.time()
        return max(0.0, ref_time - state["start"])

def get_zone_dot_color(pid):
    """Get dot color based on dwell duration thresholds"""
    duration = get_zone_dwell_duration(pid)
    for threshold, color in ZONE_THRESHOLDS:
        if duration >= threshold:
            return color
    return DEFAULT_ZONE_DOT_COLOR

def record_zone_dot_entry(prod_id, pid, timestamp):
    """Record or update a zone dot entry - dots persist for the whole day"""
    global current_day
    today = datetime.now().date()
    
    # Check if it's a new day and clear history if needed
    if today != current_day:
        with zone_dot_history_lock:
            zone_dot_history.clear()
        current_day = today
    
    duration = get_zone_dwell_duration(pid)
    color = get_zone_dot_color(pid)
    
    with zone_dot_history_lock:
        existing_dots = zone_dot_history[prod_id]
        # Find existing dot for this person in this zone today
        person_dot_idx = None
        for idx, dot in enumerate(existing_dots):
            if dot.get("person_id") == pid:
                person_dot_idx = idx
                break
        
        if person_dot_idx is not None:
            # Update existing dot with latest color (if dwell time increased)
            # Compare colors (red > orange > yellow > green)
            color_priority = {
                (0, 0, 255): 4,      # red
                (0, 165, 255): 3,    # orange
                (0, 255, 255): 2,    # yellow
                (0, 200, 0): 1       # green
            }
            last_priority = color_priority.get(tuple(existing_dots[person_dot_idx]["color"]), 0)
            current_priority = color_priority.get(tuple(color), 0)
            
            # Update if color changed to a higher priority (longer dwell)
            if current_priority > last_priority:
                existing_dots[person_dot_idx]["color"] = color
                existing_dots[person_dot_idx]["timestamp"] = timestamp
                existing_dots[person_dot_idx]["duration"] = duration
        else:
            # New dot entry for this person in this zone
            zone_dot_history[prod_id].append({
                "timestamp": timestamp,
                "color": color,
                "person_id": pid,
                "duration": duration
            })

def get_product_name(prod_id):
    """Get product name from MongoDB by product_id"""
    doc = zones_col.find_one({"product_id": prod_id})
    return doc["name"] if doc else prod_id

def update_registry(embedding):
    """Update global person registry with new embedding, return (person_id, is_new)"""
    global next_global_id
    with registry_lock:
        best_id, best_sim = None, -1.0
        for e in registry:
            sim = cosine(e["emb"], embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = e["id"]

        if best_sim >= REID_SIM_THRESHOLD:
            # Update existing person's embedding with momentum
            for e in registry:
                if e["id"] == best_id:
                    e["emb"] = l2_norm(REID_UPDATE_MOMENTUM * e["emb"] + (1 - REID_UPDATE_MOMENTUM) * embedding)
                    e["last_seen"] = time.time()
                    return best_id, False

        # New person
        nid = next_global_id
        next_global_id += 1
        registry.append({"id": nid, "emb": embedding.copy(), "last_seen": time.time()})
        return nid, True

# ------------------------------------------
# ZONE LOADING
# ------------------------------------------
def load_zones_for_camera(cam_idx):
    """Load product zones from MongoDB for a specific camera"""
    query = {"$or": [
        {"camera_id": cam_idx},
        {"camera_id": {"$exists": False}},
        {"camera_id": "all"}
    ]}
    docs = list(zones_col.find(query))
    if not docs and cam_idx != 0:
        # Fallback to camera 0 templates so demo feeds can reuse the same layout
        docs = list(zones_col.find({"camera_id": 0}))
    zones = []
    for d in docs:
        if "coordinates" not in d:
            continue

        coords = d["coordinates"]
        if len(coords) != 4:
            continue

        zones.append({
            "product_id": d.get("product_id", "unknown"),
            "name": d.get("name", "unknown"),
            "price": d.get("price", None),
            "coords": coords
        })
    return zones

# ------------------------------------------
# DETECTION + RE-ID
# ------------------------------------------
def detect_and_reid(frame):
    """Detect persons and extract ReID embeddings"""
    h, w = frame.shape[:2]
    model_h, model_w = 320, 544

    img = cv2.resize(frame, (model_w, model_h))
    inp = img.astype(np.float32).transpose(2, 0, 1)[None, :]

    dets = compiled_det([inp])[det_out]
    dets = dets.reshape(-1, 7) if dets.size else np.zeros((0, 7))

    boxes, cents, ids, emb_list = [], [], [], []

    for d in dets:
        conf = float(d[2])
        if conf < 0.55:
            continue

        x1, y1, x2, y2 = int(d[3] * w), int(d[4] * h), int(d[5] * w), int(d[6] * h)
        if x2 - x1 < 30 or y2 - y1 < 50:
            continue

        boxes.append([x1, y1, x2, y2])
        cents.append(((x1 + x2) // 2, (y1 + y2) // 2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            ids.append(-1)
            emb_list.append(None)
            continue

        reid_in = cv2.resize(crop, (128, 256)).astype(np.float32).transpose(2, 0, 1)[None, :]
        emb = l2_norm(compiled_reid([reid_in])[reid_out].flatten())

        pid, is_new = update_registry(emb)
        ids.append(pid)
        emb_list.append(emb)

    return boxes, cents, ids, emb_list

# ------------------------------------------
# HEATMAP
# ------------------------------------------
def gaussian_kernel(k):
    """Generate Gaussian kernel for heatmap"""
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    xx, yy = np.meshgrid(ax, ax)
    sigma = k / 6
    ker = np.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    return ker / ker.max()

GAUSS = gaussian_kernel(HEATMAP_KERNEL_SIZE)

def update_heatmap(cam_idx, centroid, frame_shape):
    """Add Gaussian kernel to heatmap accumulator at centroid position"""
    with heatmap_lock:
        if cam_idx not in heatmap_accumulators:
            heatmap_accumulators[cam_idx] = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)

        h, w = heatmap_accumulators[cam_idx].shape
        k_half = HEATMAP_KERNEL_SIZE // 2
        x, y = int(centroid[0]), int(centroid[1])

        # Place kernel centered at centroid
        y_start = max(0, y - k_half)
        y_end = min(h, y + k_half + 1)
        x_start = max(0, x - k_half)
        x_end = min(w, x + k_half + 1)

        kernel_y_start = max(0, k_half - y)
        kernel_y_end = kernel_y_start + (y_end - y_start)
        kernel_x_start = max(0, k_half - x)
        kernel_x_end = kernel_x_start + (x_end - x_start)

        if y_end > y_start and x_end > x_start:
            heatmap_accumulators[cam_idx][y_start:y_end, x_start:x_end] += GAUSS[
                kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end
            ]

def decay_heatmaps():
    """Periodically decay heatmap accumulators"""
    while not stop_event.is_set():
        time.sleep(2)
        with heatmap_lock:
            for cam_idx in heatmap_accumulators:
                heatmap_accumulators[cam_idx] *= HEATMAP_DECAY

def blend_heatmap(frame, cam_idx):
    """Blend heatmap overlay onto frame"""
    with heatmap_lock:
        if cam_idx not in heatmap_accumulators:
            return frame

        hmap = heatmap_accumulators[cam_idx]
        hmap_normalized = np.clip(hmap / (hmap.max() + 1e-6), 0, 1)

        # Create colored heatmap
        hmap_colored = cv2.applyColorMap((hmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        hmap_colored = cv2.resize(hmap_colored, (frame.shape[1], frame.shape[0]))

        # Blend with frame
        overlay = cv2.addWeighted(frame, 0.7, hmap_colored, 0.3, 0)
        return overlay

# ------------------------------------------
# CAMERA THREAD
# ------------------------------------------
def camera_reader(src, q, cam_idx):
    """Read frames from camera source and put into queue"""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera source: {src}")
        return

    # Initialize heatmap for this camera
    with heatmap_lock:
        heatmap_accumulators[cam_idx] = np.zeros((360, 640), dtype=np.float32)

    last_frame = None
    zones = load_zones_for_camera(cam_idx)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # Video ended - play once, then hold on last frame
            if last_frame is not None:
                with q.mutex:
                    q.queue.clear()
                q.put((last_frame.copy(), zones))
                time.sleep(FRAME_THROTTLE)
            else:
                # No frames read yet, wait a bit and try again
                time.sleep(FRAME_THROTTLE)
            continue

        frame = cv2.resize(frame, (640, 360))
        last_frame = frame.copy()  # Store last valid frame
        zones = load_zones_for_camera(cam_idx)

        with q.mutex:
            q.queue.clear()
        q.put((frame, zones))

        time.sleep(FRAME_THROTTLE)

    cap.release()

# ------------------------------------------
# ZONE MATCHING
# ------------------------------------------
def match_centroid_to_zones(c, zones):
    """Check if centroid falls inside any product zone"""
    x, y = c
    res = []
    for z in zones:
        x1, y1, x2, y2 = z["coords"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            res.append(z["product_id"])
    return res

# ------------------------------------------
# PROCESS NEW ENTRIES (Product Analytics)
# ------------------------------------------
def process_new_entries(per_cam_results):
    """Process person detections and update product analytics"""
    now = time.time()

    for cam_idx, (boxes, cents, ids, embs, zones) in enumerate(per_cam_results):
        for i, pid in enumerate(ids):
            if pid < 0:
                continue

            c = cents[i]

            # Update path tracing
            with paths_lock:
                paths[pid].append((c[0], c[1]))

            # Update heatmap
            update_heatmap(cam_idx, c, (360, 640))

            # Check zone matching
            prods = match_centroid_to_zones(c, zones)
            update_person_zone_state(pid, prods, now)
            if not prods:
                continue

            # Record zone dot entries for persistent display
            for p in prods:
                record_zone_dot_entry(p, pid, now)

            # Update product stats
            with product_stats_lock:
                for p in prods:
                    ps = product_stats.setdefault(
                        p,
                        {"footfall": 0, "dwell_sessions": [], "current_sessions": {}}
                    )

                    if pid not in ps["current_sessions"]:
                        # New visit - increment footfall
                        ps["footfall"] += 1
                        ps["current_sessions"][pid] = {"start": now, "last": now}
                    else:
                        # Update existing session
                        ps["current_sessions"][pid]["last"] = now

# ------------------------------------------
# CLOSE STALE SESSIONS
# ------------------------------------------
def close_stale_sessions():
    """Close dwelling sessions that haven't been updated recently"""
    while not stop_event.is_set():
        time.sleep(5)
        cutoff = time.time() - 10

        with product_stats_lock:
            for pid, ps in product_stats.items():
                finished = []
                for person, d in ps["current_sessions"].items():
                    if d["last"] < cutoff:
                        # Session ended
                        dwell_time = d["last"] - d["start"]
                        ps["dwell_sessions"].append(dwell_time)
                        finished.append(person)

                for f in finished:
                    ps["current_sessions"].pop(f, None)

# ------------------------------------------
# UI FRAME COMPOSITION
# ------------------------------------------
def draw_path(frame, pid, color):
    """Draw person path on frame"""
    with paths_lock:
        path_points = list(paths[pid])
        if len(path_points) < 2:
            return

        for i in range(1, len(path_points)):
            pt1 = (int(path_points[i-1][0]), int(path_points[i-1][1]))
            pt2 = (int(path_points[i][0]), int(path_points[i][1]))
            cv2.line(frame, pt1, pt2, color, 2)

def compose_tiled_frame(frames, per_cam_results):
    """Compose tiled output from multiple camera frames"""
    if not frames:
        return np.zeros((360, 640, 3), dtype=np.uint8)

    num_cams = len(frames)
    
    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(num_cams)))
    rows = int(np.ceil(num_cams / cols))
    
    h, w = frames[0].shape[:2]
    tile_h, tile_w = h, w
    
    # Create output canvas
    out_h = rows * tile_h
    out_w = cols * tile_w
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Place frames in grid
    for idx, (frame, (boxes, cents, ids, embs, zones)) in enumerate(zip(frames, per_cam_results)):
        row = idx // cols
        col = idx % cols
        
        y_start = row * tile_h
        y_end = y_start + tile_h
        x_start = col * tile_w
        x_end = x_start + tile_w
        
        # Blend heatmap onto frame
        frame_with_heatmap = blend_heatmap(frame.copy(), idx)
        
        # Get all persistent dots from today for zones in this camera
        # These dots persist even after people leave the zones
        zone_dot_colors = collections.defaultdict(list)
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        
        # Track current occupants: (product_id, person_id) -> current_color
        current_occupants = {}
        for cent, pid in zip(cents, ids):
            if pid < 0:
                continue
            zone_hits = match_centroid_to_zones(cent, zones)
            for prod in zone_hits:
                current_occupants[(prod, pid)] = get_zone_dot_color(pid)
        
        # Build dot list: use current color for active occupants, stored color for others
        with zone_dot_history_lock:
            for z in zones:
                prod_id = z["product_id"]
                if prod_id in zone_dot_history:
                    for dot_entry in zone_dot_history[prod_id]:
                        if dot_entry["timestamp"] >= today_start:
                            pid = dot_entry.get("person_id")
                            person_key = (prod_id, pid)
                            
                            # If person is currently in this zone, use their live color
                            if person_key in current_occupants:
                                zone_dot_colors[prod_id].append(current_occupants[person_key])
                            else:
                                # Person left zone, use stored color (persistent dot)
                                zone_dot_colors[prod_id].append(dot_entry["color"])
        
        # Add dots for brand new occupants not yet in history
        for (prod, pid), color in current_occupants.items():
            person_in_history = False
            with zone_dot_history_lock:
                if prod in zone_dot_history:
                    for dot_entry in zone_dot_history[prod]:
                        if dot_entry.get("person_id") == pid and dot_entry["timestamp"] >= today_start:
                            person_in_history = True
                            break
            if not person_in_history:
                zone_dot_colors[prod].append(color)

        # Draw product zones and their occupancy dots
        for z in zones:
            x1, y1, x2, y2 = z["coords"]
            cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), (200, 100, 255), 2)
            cv2.putText(frame_with_heatmap, z["name"], (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

            dot_colors = zone_dot_colors.get(z["product_id"], [])
            if dot_colors:
                dots_per_row = max(1, (x2 - x1 - 12) // 12)
                for idx_color, color in enumerate(dot_colors):
                    row_offset = idx_color // dots_per_row
                    col_offset = idx_color % dots_per_row
                    cx = x1 + 8 + col_offset * 12
                    cy = y1 + 12 + row_offset * 12
                    if cy >= y2 - 4:
                        break  # stop drawing if we run out of vertical space
                    cv2.circle(frame_with_heatmap, (int(cx), int(cy)), 5, color, -1)
                cv2.putText(
                    frame_with_heatmap,
                    f"{len(dot_colors)} in zone",
                    (x1, min(y2 - 6, y1 + 24 + ((len(dot_colors) // dots_per_row) * 12))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
        
        # Draw bounding boxes, IDs, and paths
        for i, (box, cent, pid) in enumerate(zip(boxes, cents, ids)):
            if pid < 0:
                continue
            
            x1, y1, x2, y2 = box
            
            # Draw bounding box
            cv2.rectangle(frame_with_heatmap, (x1, y1), (x2, y2), BOX_COLOR, 2)
            
            # Draw ID
            cv2.putText(frame_with_heatmap, f"ID:{pid}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 2)
            
            # Draw path
            draw_path(frame_with_heatmap, pid, BOX_COLOR)
        
        # Place frame in output
        out[y_start:y_end, x_start:x_end] = frame_with_heatmap
    
    return out

# ------------------------------------------
# STREAM GENERATOR
# ------------------------------------------
def gen_multi_stream():
    """Generate video stream from all cameras"""
    prev = time.time()
    fps = 0.0

    while True:
        frames = []
        results = []

        for q in cam_queues:
            try:
                f, zones = q.get(timeout=1)
            except:
                f = np.zeros((360, 640, 3), dtype=np.uint8)
                zones = []
            frames.append(f)

            boxes, cents, ids, embs = detect_and_reid(f)
            results.append((boxes, cents, ids, embs, zones))

        process_new_entries(results)

        now = time.time()
        if now - prev > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev))
        prev = now

        out = compose_tiled_frame(frames, results)
        
        # Add FPS text
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

# ------------------------------------------
# MONGODB WORKERS
# ------------------------------------------
def hourly_worker():
    """Save hourly snapshot of product analytics"""
    while not stop_event.is_set():
        time.sleep(3600)  # Run every hour
        try:
            now = datetime.now()
            hour_start = now.replace(minute=0, second=0, microsecond=0)
            
            with product_stats_lock:
                snapshot = {}
                for pid, ps in product_stats.items():
                    dwell_times = ps["dwell_sessions"]
                    avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0
                    
                    snapshot[pid] = {
                        "footfall": ps["footfall"],
                        "dwell_count": len(dwell_times),
                        "avg_dwell": avg_dwell
                    }
                
                hourly_col.insert_one({
                    "timestamp": hour_start,
                    "products": snapshot
                })
                
                print(f"[HOURLY] Saved snapshot for {hour_start}")
        except Exception as e:
            print(f"[ERROR] Hourly worker failed: {e}")

def daily_worker():
    """Save daily analytics and reset stats"""
    while not stop_event.is_set():
        # Wait until midnight
        now = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        wait_seconds = (next_midnight - now).total_seconds()
        
        time.sleep(wait_seconds)
        
        try:
            now = datetime.now()
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            with product_stats_lock:
                daily_snapshot = {}
                for pid, ps in product_stats.items():
                    dwell_times = ps["dwell_sessions"]
                    avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0
                    
                    daily_snapshot[pid] = {
                        "footfall": ps["footfall"],
                        "dwell_count": len(dwell_times),
                        "avg_dwell": avg_dwell
                    }
                
                daily_col.insert_one({
                    "timestamp": day_start,
                    "products": daily_snapshot
                })
                
                # Reset stats
                product_stats.clear()
                
                # Clear zone dot history for new day
                with zone_dot_history_lock:
                    zone_dot_history.clear()
                
                print(f"[DAILY] Saved daily snapshot for {day_start} and reset stats")
        except Exception as e:
            print(f"[ERROR] Daily worker failed: {e}")

# ------------------------------------------
# ROUTES
# ------------------------------------------
@app.route("/")
def index():
    """Main dashboard page"""
    return render_template_string("""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Product Analytics Dashboard</title>
    <style>
      body { 
        background: #0e0f11; 
        color: #e8eef0; 
        font-family: Arial, sans-serif; 
        margin: 0; 
        padding: 16px; 
      }
      .wrap { 
        display: grid; 
        grid-template-columns: 2fr 1fr; 
        gap: 16px; 
      }
      .video { 
        background: #111; 
        padding: 8px; 
        border-radius: 8px; 
      }
      .card { 
        background: #121214; 
        padding: 12px; 
        border-radius: 8px; 
        margin-bottom: 12px; 
      }
      h2 { 
        margin: 6px 0 12px 0; 
        color: #00d4a6; 
      }
      table { 
        width: 100%; 
        border-collapse: collapse; 
        color: #ddd; 
      }
      td, th { 
        padding: 8px; 
        text-align: left; 
        border-bottom: 1px solid #222; 
      }
      th {
        color: #00d4a6;
        font-weight: bold;
      }
      .product-item {
        padding: 4px 0;
        border-bottom: 1px solid #333;
      }
      #video-container {
        position: relative;
        width: 100%;
      }
      img {
        width: 100%;
        border-radius: 6px;
      }
    </style>
    </head>
    <body>
      <h2>Product Analytics Dashboard</h2>

      <div class="wrap">

        <!-- LIVE VIDEO PANEL -->
        <div class="video card">
          <div id="video-container">
            <img src="/video_feed" alt="Live Video Feed">
          </div>
        </div>

        <!-- RIGHT PANEL -->
        <div>

          <!-- TOP 5 -->
          <div class="card">
            <strong>Top 5 Visited Products</strong>
            <div id="top5"></div>
          </div>

          <!-- BOTTOM 5 -->
          <div class="card">
            <strong>Least Visited Products (Bottom 5)</strong>
            <div id="low5"></div>
          </div>

          <!-- PRODUCT TABLE -->
          <div class="card">
            <strong>Live Product Table</strong>
            <table id="ptable">
              <thead>
                <tr>
                  <th>Product Name</th>
                  <th>Footfall</th>
                  <th>Avg Dwell (s)</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>

        </div>
      </div>

      <script>
        async function refresh() {
          try {
            let res = await fetch('/products/live'); 
            let data = await res.json();

            // Convert products object to array with names
            let arr = Object.keys(data.products).map(k => ({
                id: k,
                name: data.products[k].name || k,
                foot: data.products[k].footfall || 0,
                avg: data.products[k].avg_dwell || 0
            }));

            // Sort by footfall
            arr.sort((a, b) => b.foot - a.foot);

            // Top 5
            let top = arr.slice(0, 5);
            let tdiv = document.getElementById('top5');
            tdiv.innerHTML = "";
            top.forEach(x => {
                let div = document.createElement('div');
                div.className = 'product-item';
                div.innerHTML = `<strong>${x.name}</strong> — ${x.foot} visits`;
                tdiv.appendChild(div);
            });

            // Bottom 5
            let low = arr.slice(-5).reverse();
            let ldiv = document.getElementById('low5');
            ldiv.innerHTML = "";
            if (low.length > 0) {
                low.forEach(x => {
                    let div = document.createElement('div');
                    div.className = 'product-item';
                    div.innerHTML = `<strong>${x.name}</strong> — ${x.foot} visits`;
                    ldiv.appendChild(div);
                });
            } else {
                ldiv.innerHTML = "<div>No data available</div>";
            }

            // Product table
            let tbody = document.querySelector('#ptable tbody');
            tbody.innerHTML = "";
            arr.forEach(x => {
                let row = document.createElement('tr');
                row.innerHTML = `
                    <td><strong>${x.name}</strong></td>
                    <td>${x.foot}</td>
                    <td>${x.avg.toFixed(1)}</td>
                `;
                tbody.appendChild(row);
            });

          } catch(e) { 
            console.error('Error refreshing data:', e); 
          }
        }

        setInterval(refresh, 3000);
        refresh();
      </script>

    </body>
    </html>
    """)

@app.route("/video_feed")
def video_feed():
    """Video streaming endpoint"""
    return Response(gen_multi_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/products/live")
def products_live():
    """Return live product analytics as JSON"""
    out = {"products": {}, "total_unique": 0, "metrics": {}}
    
    with product_stats_lock:
        for pid, ps in product_stats.items():
            dwell_times = ps["dwell_sessions"]
            avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0
            
            product_name = get_product_name(pid)
            
            out["products"][pid] = {
                "name": product_name,
                "footfall": ps["footfall"],
                "dwell_count": len(dwell_times),
                "avg_dwell": avg_dwell
            }
    
    # Count total unique persons
    with registry_lock:
        out["total_unique"] = len(registry)
    
    return jsonify(out)

# ------------------------------------------
# APP START
# ------------------------------------------
if __name__ == "__main__":
    # Create directories if needed
    if not os.path.exists("videos"):
        os.makedirs("videos")
        print("[INFO] Created 'videos' directory")

    if not os.path.exists("models"):
        os.makedirs("models")
        print("[INFO] Created 'models' directory")

    # Check if zones exist in database
    total_zones = zones_col.count_documents({})
    if total_zones == 0:
        print("[WARNING] No product zones found in MongoDB!")
        print("[WARNING] Please add zones to the 'zones' collection in the 'product_analytics' database.")
        print("[WARNING] Expected schema: {camera_id: int, product_id: string, name: string, price: number, coordinates: [x1, y1, x2, y2]}")
    else:
        print(f"[INFO] Found {total_zones} product zones in database")

    # Start camera threads
    print(f"[INFO] Starting {len(CAM_SOURCES)} camera thread(s)...")
    for i, src in enumerate(CAM_SOURCES):
        q = queue.Queue(maxsize=1)
        cam_queues.append(q)
        t = threading.Thread(target=camera_reader, args=(src, q, i), daemon=True)
        t.start()
        cam_threads.append(t)
        print(f"[INFO] Started camera {i}: {src}")

    # Start background workers
    threading.Thread(target=close_stale_sessions, daemon=True).start()
    threading.Thread(target=decay_heatmaps, daemon=True).start()
    threading.Thread(target=hourly_worker, daemon=True).start()
    threading.Thread(target=daily_worker, daemon=True).start()
    
    print("[INFO] Background workers started")
    print("[INFO] Starting Flask server on http://0.0.0.0:5000")
    
    app.run(host="0.0.0.0", port=5000, threaded=True)
