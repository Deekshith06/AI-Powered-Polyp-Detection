import os, time, json, sys

# --- HACK: FIX STREAMLIT CLOUD OPENCV ISSUE ---
# Streamlit's Linux containers lack libgthread. 
# Ultralytics forces the install of the GUI 'opencv-python' anyways.
# This catches the libgthread ImportError and neatly replaces it.
try:
    import cv2
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless"], check=False)
    import cv2
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(page_title="Polyp AI Screen", layout="wide", initial_sidebar_state="expanded")

# --- CSS Injection ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- State Initialization ---
st.session_state.setdefault('is_playing', True)
st.session_state.setdefault('uncertain', [])
st.session_state.setdefault('no_det_frames', 0)
st.session_state.setdefault('thresh', 0.45)
st.session_state.setdefault('src', '0')

# --- Core Logic & Helpers ---
@st.cache_resource(show_spinner="Loading Engine...")
def load_core():
    from ultralytics import YOLO # Lazy load to save RAM and startup time 
    model = YOLO("yolov8n.pt")
    model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False) # Warmup
    return model

def load_corrections():
    os.makedirs("data", exist_ok=True)
    return json.load(open("data/user_corrections.json")) if os.path.exists("data/user_corrections.json") else []

def save_correction(data):
    c = load_corrections()
    with open("data/user_corrections.json", "w") as f:
        json.dump(c + [data], f)

@st.cache_resource(show_spinner=False)
def get_severity_model(_dummy_trigger=0):
    # Dummy data baseline
    X, y = [[0.5, 1000], [0.9, 5000], [0.7, 2500], [0.3, 500], [0.95, 10000]], [3, 8, 6, 2, 9]
    for c in load_corrections():
        b = c.get("bbox", [0, 0, 0, 0])
        X.append([c.get("confidence", 0.5), (b[2]-b[0]) * (b[3]-b[1])])
        y.append(c.get("severity", 5))
    return LinearRegression().fit(X, y)

def run_inference(frame, model, min_conf):
    t0 = time.time()
    res = model(frame, conf=min_conf, verbose=False)[0]
    dets, draw = [], frame.copy()
    
    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().tolist())
        c = float(b.conf[0].cpu())
        dets.append({"bbox": [x1, y1, x2, y2], "confidence": c})
        
        # Color scale
        clr = (0, 255, 0) if c > 0.85 else (255, 255, 0) if c > 0.70 else (255, 0, 0)
        cv2.rectangle(draw, (x1, y1), (x2, y2), clr, 2)
        cv2.putText(draw, f"{c:.2f}", (x1, max(y1-5, 0)), 0, 0.5, clr, 2)
        
    return draw, dets, (time.time() - t0) * 1000

# --- UI Setup ---
with st.sidebar:
    st.title("⚙️ AI Controls")
    
    # Input logic
    src_type = st.radio("Source", ["Sample video", "Live Webcam"])
    new_src = "data/sample_colonoscopy.mp4" if src_type == "Sample video" else "0"
    if new_src != st.session_state.src:
        st.session_state.src = new_src
        st.session_state.is_playing = True
        
    # Threshold logic
    auto = st.checkbox("Auto-Adjust Threshold", value=False)
    if auto:
        st.info(f"Auto threshold: {st.session_state.thresh:.2f}")
    else:
        st.session_state.thresh = st.slider("Threshold", 0.0, 1.0, st.session_state.thresh, 0.05)
        
    st.session_state.is_playing = not st.button("▶️ Play / ⏸️ Pause Stream") if st.session_state.is_playing else st.button("▶️ Play / ⏸️ Pause Stream")

    st.markdown("---")
    if st.button("Retrain Predictor"):
        st.session_state.severity_model_trigger = time.time() # Force reload
        
    st.download_button("Export Corrections", json.dumps(load_corrections()), "corrections.json", "application/json")


# --- Main Screen ---
st.title("🔬 Real-time Polyp Detection")
c1, c2 = st.columns([3, 1])
video_box, fps_box, det_box, lat_box, sev_box = c1.empty(), c2.empty(), c2.empty(), c2.empty(), c2.empty()

# --- Feedback Loop UI ---
fb_box = st.empty()
def render_fb():
    with fb_box.container():
        st.subheader("👨‍⚕️ Clinician Feedback")
        for i, u in enumerate(st.session_state.uncertain):
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.write(f"Detection {i+1} (Conf: {u['confidence']:.2f})")
            if col2.button("✔️ Confirm", key=f"c{i}_{time.time()}"):
                save_correction({"confidence": u["confidence"], "bbox": u["bbox"], "severity": 8})
                st.session_state.uncertain.pop(i)
                if not st.session_state.is_playing: st.rerun()
            if col3.button("❌ Reject", key=f"r{i}_{time.time()}"):
                save_correction({"confidence": u["confidence"], "bbox": u["bbox"], "severity": 1})
                st.session_state.uncertain.pop(i)
                if not st.session_state.is_playing: st.rerun()
                
if not st.session_state.is_playing:
    render_fb()
    video_box.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="RGB")
    
# --- Action Loop ---
if st.session_state.is_playing:
    model = load_core()
    sev_model = get_severity_model(st.session_state.get('severity_model_trigger', 0))
    cap = cv2.VideoCapture(int(st.session_state.src) if st.session_state.src.isdigit() else st.session_state.src)
    
    while st.session_state.is_playing:
        t_loop = time.time()
        
        # Pull 3 frames to keep buffer fresh but only process 1
        for _ in range(3): ret, f = cap.read()
        
        if not ret or f is None:
            # Reconnect/Loop
            if str(st.session_state.src).isdigit():
                time.sleep(0.5); cap = cv2.VideoCapture(int(st.session_state.src)); continue
            else:
                if not os.path.exists(st.session_state.src):
                    st.error(f"Sample video not found at {st.session_state.src}. Please download it or switch to Webcam.")
                    st.session_state.is_playing = False
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, f = cap.read()
                if not ret: break

        frame = cv2.cvtColor(cv2.resize(f, (640, int(f.shape[0]*(640/f.shape[1])))), cv2.COLOR_BGR2RGB)
        
        # Inference
        res_frame, dets, t_inf = run_inference(frame, model, st.session_state.thresh)
        video_box.image(res_frame, channels="RGB")
        
        # Stats
        fps = 1.0 / (time.time() - t_loop)
        fps_box.metric("FPS", f"{fps:.1f}")
        det_box.metric("Polyps", len(dets))
        lat_box.metric("Latency", f"{t_inf:.1f} ms")
        
        # Auto-Adjust
        if auto:
            if not dets:
                st.session_state.no_det_frames += 1
                if st.session_state.no_det_frames > 10:
                    st.session_state.thresh = max(0.05, st.session_state.thresh - 0.05)
                    st.session_state.no_det_frames = 0
            else:
                st.session_state.no_det_frames = 0
                if len(dets) > 10:
                    st.session_state.thresh = min(0.60, st.session_state.thresh + 0.10)
                    
        # Severity
        if dets:
            top = max(dets, key=lambda x: x["confidence"])
            w, h = top["bbox"][2]-top["bbox"][0], top["bbox"][3]-top["bbox"][1]
            sev = max(1, min(10, int(round(sev_model.predict([[top["confidence"], w*h]])[0]))))
            clr = "#ef4444" if sev > 7 else "#f59e0b" if sev > 4 else "#10b981"
            sev_box.markdown(f"<div style='background:rgba(255,255,255,0.05);padding:10px;text-align:center;border-radius:10px;'><h3 style='color:#94a3b8;margin:0;'>Severity</h3><h1 style='color:{clr};margin:0;'>{sev}/10</h1></div>", unsafe_allow_html=True)
            
            # Catch uncertain
            if 0.3 <= top["confidence"] <= 0.65 and len(st.session_state.uncertain) < 5:
                if not any(np.allclose(u["bbox"], top["bbox"], atol=30) for u in st.session_state.uncertain):
                    st.session_state.uncertain.append(top)
                    render_fb()
        else:
            sev_box.markdown("<div style='text-align:center;color:#94a3b8;'>No polyps</div>", unsafe_allow_html=True)
            
        time.sleep(0.01) # UI rest
