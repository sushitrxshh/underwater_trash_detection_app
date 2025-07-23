import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import uuid
from ultralytics import YOLO
from streamlit_webrtc import webrtc_stream_recorder, WebRtcMode, VideoProcessorBase
import av # For video processing in streamlit-webrtc
import queue # For frame processing queue in streamlit-webrtc

# --- Trash Classes (Embedded from trash_classes.py) ---
# This file contains the class names for the actual 15 classes the model was trained on

# FLEXIBLE MAPPING: This will be updated based on actual model behavior
# The issue is that class IDs from the model don't match our assumption

# Expected class names in the order you provided
EXPECTED_CLASSES = ["mask", "can", "cellphone", "electronics", "gbottle", "glove", "metal", "misc", "net", "pbag", "pbottle", "plastic", "rod", "sunglasses", "tyre"]

# Display names for better readability
DISPLAY_NAMES = ["Mask", "Can", "Cellphone", "Electronics", "Glass Bottle", "Glove", "Metal", "Misc", "Net", "Plastic Bag", "Plastic Bottle", "Plastic", "Rod", "Sunglasses", "Tyre"]

# Colors for different trash types (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),       # Green for mask
    (255, 0, 0),       # Blue for can
    (255, 255, 0),     # Cyan for cellphone
    (0, 0, 255),       # Red for electronics
    (255, 255, 255),   # White for glass bottle
    (0, 255, 255),     # Yellow for glove
    (128, 128, 128),   # Gray for metal
    (255, 0, 255),     # Magenta for misc
    (0, 128, 255),     # Orange for net
    (0, 255, 128),     # Light green for plastic bag
    (128, 0, 128),     # Purple for plastic bottle
    (255, 128, 0),     # Light blue for plastic
    (128, 0, 0),       # Dark blue for rod
    (255, 255, 128),   # Light yellow for sunglasses
    (0, 128, 0)        # Dark green for tyre
]

def get_class_name(class_id):
    """Get the class name for a given class ID"""
    if 0 <= class_id < len(EXPECTED_CLASSES):
        return EXPECTED_CLASSES[class_id]
    return f"Unknown_{class_id}"

def get_class_name_short(class_id):
    """Get the short class name for a given class ID"""
    if 0 <= class_id < len(DISPLAY_NAMES):
        return DISPLAY_NAMES[class_id]
    return f"Unknown_{class_id}"

def get_class_color(class_id):
    """Get the color for a given class ID"""
    if 0 <= class_id < len(COLORS):
        return COLORS[class_id]
    return (0, 255, 0)  # Default green

def get_all_classes():
    """Get all class names"""
    return {i: name for i, name in enumerate(EXPECTED_CLASSES)}

def get_all_classes_short():
    """Get all short class names"""
    return {i: name for i, name in enumerate(DISPLAY_NAMES)}

def update_mapping_from_model(model_names_dict):
    """Update mapping based on actual model class names from a dictionary"""
    global EXPECTED_CLASSES, DISPLAY_NAMES
    if model_names_dict:
        # Sort by key (class ID) to ensure consistent order
        sorted_names = sorted(model_names_dict.items())
        EXPECTED_CLASSES = [name for _, name in sorted_names]
        DISPLAY_NAMES = [name.title() for _, name in sorted_names]
        st.info(f"Updated class mapping from model: {EXPECTED_CLASSES}")
    else:
        st.warning("Model class names not found or empty. Using default mapping.")
        # Optionally reset to a known default if model names are critical and missing
        # For now, keep existing EXPECTED_CLASSES/DISPLAY_NAMES if model_names_dict is empty

# --- YOLO Model Loading (Cached) ---
@st.cache_resource
def load_yolo_model_cached():
    model_path = 'best.pt'
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found! Please ensure 'best.pt' is in the same directory as this script.")
        return None
    try:
        model = YOLO(model_path)
        if hasattr(model, 'names') and model.names:
            update_mapping_from_model(model.names)
        else:
            st.warning("Model has no class names. Using default mapping.")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.exception(e)
        return None

# --- Video Processing Function ---
def process_video_file(video_path, model, frame_skip, confidence_threshold, max_detections):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Could not open video file. Please check if the file is valid.")
        return None, None, None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary files for output videos
    temp_original_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_detected_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_original_output.close()
    temp_detected_output.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4, widely supported
    out_detected = cv2.VideoWriter(temp_detected_output.name, fourcc, fps, (width, height))
    out_original = cv2.VideoWriter(temp_original_output.name, fourcc, fps, (width, height))

    if not out_detected.isOpened() or not out_original.isOpened():
        st.error("Could not initialize video writers. Check codecs and permissions.")
        cap.release()
        return None, None, None

    frame_count = 0
    processed_frames_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write original frame to its output video
        out_original.write(frame)

        if frame_count % frame_skip == 0:
            processed_frame = frame.copy()
            results = model(frame)
            
            detection_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        if detection_count >= max_detections:
                            break
                            
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        if confidence < confidence_threshold:
                            continue
                        
                        class_name = get_class_name_short(class_id)
                        color = get_class_color(class_id)
                        
                        cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(processed_frame, label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        detection_count += 1
            
            out_detected.write(processed_frame)
            processed_frames_count += 1
        
        frame_count += 1
        progress_bar.progress(min(1.0, frame_count / total_frames))
        status_text.text(f"Processing frame {frame_count}/{total_frames}...")

    cap.release()
    out_detected.release()
    out_original.release()
    
    progress_bar.empty()
    status_text.empty()

    return temp_original_output.name, temp_detected_output.name, {
        'width': width, 'height': height, 'fps': fps,
        'total_frames': total_frames, 'processed_frames_count': processed_frames_count
    }

# --- Streamlit-WebRTC Video Processor Class ---
# This class processes frames from the webcam stream
class VideoTransformer(VideoProcessorBase):
    def __init__(self, model, confidence_threshold, max_detections):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.frame_queue = queue.Queue(maxsize=1) # To limit processing rate

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.frame_queue.full():
            return frame # Skip if queue is full (previous frame still processing)

        img = frame.to_ndarray(format="bgr24") # Convert av.VideoFrame to numpy array (BGR)

        # Put frame in queue for asynchronous processing
        self.frame_queue.put(img)

        # Get processed frame from cache if available
        processed_img = self._process_frame_cached(img)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    # This method is called by recv and processes the frame
    # We use a simple cache to avoid re-processing the same frame if recv is called too fast
    @st.cache_data(ttl=0.1) # Cache for 0.1 seconds to reduce redundant processing
    def _process_frame_cached(self, img):
        # Run inference on the frame
        results = self.model(img, verbose=False) # verbose=False to suppress console output

        processed_frame = img.copy()
        
        detection_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if detection_count >= self.max_detections:
                        break
                        
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence < self.confidence_threshold:
                        continue
                    
                    class_name = get_class_name_short(class_id)
                    color = get_class_color(class_id)
                    
                    cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(processed_frame, label, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detection_count += 1
        
        # Clear the queue after processing this frame
        if not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        return processed_frame

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Underwater Trash Detection",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (similar to previous style.css)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lato:wght@300;400;700&family=Poppins:wght@300;400;600;700&display=swap');

    :root {
        --color-deep-ocean: #001a33;
        --color-dark-water: #003366;
        --color-mid-ocean: #004d99;
        --color-light-ocean: #006994;
        --color-seafoam: #87ceeb;
        --color-white: #ffffff;
        --color-text-primary: var(--color-white);
        --color-text-secondary: rgba(255, 255, 255, 0.8);
        --gradient-background: linear-gradient(135deg, var(--color-light-ocean) 0%, var(--color-dark-water) 50%, var(--color-deep-ocean) 100%);
        --card-bg-blur: rgba(255, 255, 255, 0.08);
        --card-backdrop-blur: 20px;
        --card-border-light: rgba(255, 255, 255, 0.15);
        --card-shadow-outer: 0 15px 35px rgba(0, 0, 0, 0.4);
        --card-shadow-inner: inset 0 0 10px rgba(0, 0, 0, 0.2);
        --font-heading: 'Poppins', sans-serif;
        --font-body: 'Lato', sans-serif;
    }

    body {
        font-family: var(--font-body);
        color: var(--color-text-secondary);
    }

    .stApp {
        background: var(--gradient-background);
        animation: backgroundShift 20s ease-in-out infinite alternate;
    }

    @keyframes backgroundShift {
        0%, 100% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
    }

    /* Header */
    .st-emotion-cache-18ni7ap { /* Main App Header container */
        background: transparent !important;
    }
    .st-emotion-cache-1r6y40z { /* Header */
        background: transparent !important;
    }
    .st-emotion-cache-10qj07x { /* Header content */
        background: transparent !important;
    }
    .st-emotion-cache-k3gqf4 { /* Header title */
        font-family: var(--font-heading);
        font-size: 3.5rem; /* Adjusted for Streamlit header */
        font-weight: 700;
        text-shadow: 4px 4px 10px rgba(0,0,0,0.8);
        background: linear-gradient(45deg, var(--color-white), var(--color-seafoam));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        text-align: center;
    }
    .st-emotion-cache-10qj07x p { /* Header tagline */
        font-family: var(--font-body);
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 300;
        text-align: center;
        color: var(--color-text-secondary);
    }

    /* Sidebar */
    .st-emotion-cache-12fmj7g { /* Sidebar container */
        background: rgba(255, 255, 255, 0.05); /* Sidebar background */
        backdrop-filter: blur(var(--card-backdrop-blur));
        border-right: 1px solid var(--card-border-light);
        box-shadow: none;
        padding: 30px;
    }
    .st-emotion-cache-12fmj7g .st-emotion-cache-10qj07x { /* Sidebar sections */
        padding-top: 20px;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .st-emotion-cache-12fmj7g .st-emotion-cache-10qj07x:last-child {
        border-bottom: none;
    }
    .st-emotion-cache-12fmj7g h2 { /* Sidebar title */
        font-family: var(--font-heading);
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--color-seafoam);
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 30px;
    }
    .st-emotion-cache-12fmj7g h3 { /* Sidebar section titles */
        font-family: var(--font-heading);
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--color-white);
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 20px;
    }
    .st-emotion-cache-12fmj7g h3 .icon { /* Font Awesome icon in sidebar titles */
        color: var(--color-seafoam);
        font-size: 1.2em;
    }

    /* Main Content Cards */
    .st-emotion-cache-10qj07x > div > .st-emotion-cache-10qj07x:not(.st-emotion-cache-1r6y40z) { /* Target main content cards */
        background: var(--card-bg-blur);
        backdrop-filter: blur(var(--card-backdrop-blur));
        border-radius: 25px;
        padding: 40px;
        border: 1px solid var(--card-border-light);
        box-shadow: var(--card-shadow-outer), var(--card-shadow-inner);
        transition: transform 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-bottom: 40px; /* Space between cards */
    }
    .st-emotion-cache-10qj07x > div > .st-emotion-cache-10qj07x:not(.st-emotion-cache-1r6y40z):hover {
        transform: translateY(-10px) scale(1.01);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6), var(--card-shadow-inner);
    }
    .st-emotion-cache-10qj07x h2:first-child { /* Card titles */
        font-family: var(--font-heading);
        font-size: 2rem;
        color: var(--color-seafoam);
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
    }
    .st-emotion-cache-10qj07x h2:first-child .icon {
        font-size: 1.8em;
        color: var(--color-seafoam);
    }

    /* Buttons */
    .st-emotion-cache-10qj07x button {
        background: linear-gradient(45deg, #007bff, #00bfff);
        color: var(--color-white);
        border: none;
        padding: 12px 25px;
        border-radius: 30px;
        cursor: pointer;
        font-family: var(--font-heading);
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 15px;
    }
    .st-emotion-cache-10qj07x button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 123, 255, 0.5);
    }
    .st-emotion-cache-10qj07x button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        background: linear-gradient(45deg, #6c757d, #9c9c9c);
        transform: none;
        box-shadow: none;
    }

    /* File Uploader */
    .st-emotion-cache-10qj07x .st-emotion-cache-10qj07x div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(255, 255, 255, 0.5);
        border-radius: 18px;
        padding: 40px 30px;
        background: rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .st-emotion-cache-10qj07x .st-emotion-cache-10qj07x div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--color-seafoam);
        background: rgba(135, 206, 235, 0.1);
    }
    .st-emotion-cache-10qj07x .st-emotion-cache-10qj07x div[data-testid="stFileUploaderDropzone"] p {
        font-family: var(--font-heading);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--color-white);
    }
    .st-emotion-cache-10qj07x .st-emotion-cache-10qj07x div[data-testid="stFileUploaderDropzone"] svg {
        color: var(--color-seafoam);
        font-size: 3em;
        margin-bottom: 10px;
    }

    /* Sliders */
    .st-emotion-cache-10qj07x .st-emotion-cache-10qj07x div[data-testid="stSlider"] label {
        font-family: var(--font-heading);
        font-size: 1rem;
        font-weight: 600;
        color: var(--color-seafoam);
    }
    .st-emotion-cache-10qj07x .st-emotion-cache-10qj07x div[data-testid="stSlider"] div[role="slider"] {
        background: rgba(255, 255, 255, 0.2);
        height: 8px;
        border-radius: 4px;
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.3);
    }
    .st-emotion-cache-10qj07x .st-emotion-cache-10qj07x div[data-testid="stSlider"] div[role="slider"] > div:first-child {
        background: var(--color-seafoam);
        border: 3px solid var(--color-white);
        width: 22px;
        height: 22px;
        border-radius: 50%;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        top: -7px; /* Adjust thumb position */
    }

    /* Expander (Debug Info) */
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] {
        background: var(--card-bg-blur);
        backdrop-filter: blur(var(--card-backdrop-blur));
        border-radius: 25px;
        padding: 30px; /* Adjusted padding for expander */
        border: 1px solid var(--card-border-light);
        box-shadow: var(--card-shadow-outer), var(--card-shadow-inner);
        transition: transform 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        margin-bottom: 40px;
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"]:hover {
        transform: translateY(-10px) scale(1.01);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6), var(--card-shadow-inner);
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] button { /* Expander header button */
        font-family: var(--font-heading);
        font-size: 1.8rem; /* Adjusted for expander title */
        color: var(--color-seafoam);
        font-weight: 700;
        text-align: left;
        padding: 0;
        margin: 0;
        background: transparent;
        box-shadow: none;
        text-transform: none;
        letter-spacing: normal;
        justify-content: flex-start;
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] button:hover {
        transform: none;
        box-shadow: none;
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] button .icon {
        font-size: 1.5em;
        margin-right: 15px;
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] { /* Expander content */
        padding-top: 25px;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 25px;
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] h4 { /* Debug group titles */
        font-family: var(--font-heading);
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--color-white);
        margin-bottom: 15px;
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] p { /* Debug info lines */
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.95rem;
        margin-bottom: 8px;
        color: var(--color-text-secondary);
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] .debug-label {
        font-weight: 600;
        color: var(--color-seafoam);
        min-width: 150px;
        text-align: left;
    }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] .true-status { color: #00ff00; font-weight: 700; }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] .false-status { color: #ff0000; font-weight: 700; }
    .st-emotion-cache-10qj07x div[data-testid="stExpander"] .unknown-status { color: #ffd700; font-weight: 700; }

    /* Video Players */
    .st-emotion-cache-10qj07x div[data-testid="stVideo"] { /* Container for st.video */
        background: rgba(0,0,0,0.8);
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    .st-emotion-cache-10qj07x div[data-testid="stVideo"] video {
        border-radius: 15px;
    }

    /* Columns for side-by-side video */
    .st-emotion-cache-10qj07x div[data-testid="stHorizontalBlock"] {
        gap: 30px; /* Space between columns */
    }
    .st-emotion-cache-10qj07x div[data-testid="stHorizontalBlock"] > div {
        flex: 1; /* Make columns take equal width */
    }
    .st-emotion-cache-10qj07x div[data-testid="stHorizontalBlock"] h4 { /* Video titles */
        font-family: var(--font-heading);
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--color-seafoam);
        margin-bottom: 15px;
        text-align: center;
    }

    /* Streamlit Status (Spinner, Success, Error) */
    .st-emotion-cache-10qj07x div[data-testid="stStatusContainer"] {
        background: var(--card-bg-blur);
        backdrop-filter: blur(var(--card-backdrop-blur));
        border-radius: 15px;
        padding: 20px;
        border: 1px solid var(--card-border-light);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        margin-top: 30px;
    }
    .st-emotion-cache-10qj07x div[data-testid="stStatusContainer"] p {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--color-white);
    }
    .st-emotion-cache-10qj07x div[data-testid="stStatusContainer"] svg {
        color: var(--color-seafoam);
    }

    /* General text within cards */
    .st-emotion-cache-10qj07x p {
        color: var(--color-text-secondary);
    }
    .st-emotion-cache-10qj07x small {
        font-size: 0.85rem;
        opacity: 0.8;
    }

    /* Hide Streamlit default header/footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; } /* Hide default top header */
</style>
""", unsafe_allow_html=True)

# --- App Title and Tagline ---
st.markdown(
    """
    <h1 class="st-emotion-cache-k3gqf4">🌊 Advanced Underwater Trash Detection</h1>
    <p class="st-emotion-cache-10qj07x">Protecting our oceans, one precise detection at a time</p>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Controls ---
st.sidebar.markdown(
    """
    <h2>Controls</h2>
    """,
    unsafe_allow_html=True
)

# Model Status Section
st.sidebar.markdown(
    """
    <h3><i class="fas fa-microchip icon"></i> Model Status</h3>
    """,
    unsafe_allow_html=True
)
model_status_indicator_placeholder = st.sidebar.empty()
model_classes_count_placeholder = st.sidebar.empty()
reload_model_button = st.sidebar.button("Reload Model", key="reload_model_btn")

# Detection Settings Section
st.sidebar.markdown(
    """
    <h3><i class="fas fa-sliders-h icon"></i> Detection Settings</h3>
    """,
    unsafe_allow_html=True
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold:",
    min_value=0.1, max_value=0.9, value=0.6, step=0.05,
    format="%.2f"
)
frame_skip_interval = st.sidebar.slider(
    "Frame Skip Interval:",
    min_value=1, max_value=20, value=10, step=1
)
max_detections = st.sidebar.slider(
    "Max Detections per Frame:",
    min_value=1, max_value=50, value=20, step=1
)

# Refresh Status Button in Sidebar
st.sidebar.markdown("---") # Separator
refresh_status_button = st.sidebar.button("Refresh Status", key="refresh_status_btn")

# --- Main Content Area ---
main_placeholder = st.empty() # Placeholder for global status messages
main_status_message = st.empty() # Placeholder for specific process messages

# --- Load Model ---
model = load_yolo_model_cached()

# --- Update Model Status UI ---
def update_model_status_ui():
    if model:
        model_status_indicator_placeholder.markdown(
            f'<div class="model-status-info"><span class="status-indicator model-loaded"><i class="fas fa-check-circle"></i></span> <span>Model Loaded</span></div>',
            unsafe_allow_html=True
        )
        model_classes_count_placeholder.markdown(f"Classes: {len(model.names) if hasattr(model, 'names') else 'N/A'}")
    else:
        model_status_indicator_placeholder.markdown(
            f'<div class="model-status-info"><span class="status-indicator model-unloaded"><i class="fas fa-times-circle"></i></span> <span>Model Unloaded</span></div>',
            unsafe_allow_html=True
        )
        model_classes_count_placeholder.markdown("Classes: 0")

update_model_status_ui()

# --- Reload Model Logic ---
if reload_model_button:
    st.cache_resource.clear() # Clear the cache for the model
    model = load_yolo_model_cached() # Attempt to reload
    update_model_status_ui()
    if model:
        main_placeholder.success("✅ Model reloaded successfully!")
    else:
        main_placeholder.error("❌ Failed to reload model. Check logs.")

# --- Debug Information Expander ---
with st.expander("<h3><i class='fas fa-bug icon'></i> Debug Information</h3>", unsafe_allow_html=True):
    if st.button("Refresh Debug Info", key="refresh_debug_info_btn_expander"):
        # This button also effectively re-runs the model loading and updates UI
        update_model_status_ui() # Update sidebar status
        st.experimental_rerun() # Rerun app to refresh all debug info

    st.markdown("<h4>Environment Check:</h4>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='debug-label'>OpenCV Available:</span> <span class='{'true-status' if 'cv2' in sys.modules else 'false-status'}'>{'True' if 'cv2' in sys.modules else 'False'}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='debug-label'>YOLO Library Available:</span> <span class='{'true-status' if 'ultralytics' in sys.modules else 'false-status'}'>{'True' if 'ultralytics' in sys.modules else 'False'}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='debug-label'>Trash Classes Imported:</span> <span class='{'true-status' if 'EXPECTED_CLASSES' in globals() else 'false-status'}'>{'True' if 'EXPECTED_CLASSES' in globals() else 'False'}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='debug-label'>Model File Exists:</span> <span class='{'true-status' if os.path.exists('best.pt') else 'false-status'}'>{'True' if os.path.exists('best.pt') else 'False'}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='debug-label'>Model Loaded:</span> <span class='{'true-status' if model else 'false-status'}'>{'True' if model else 'False'}</span></p>", unsafe_allow_html=True)

    st.markdown("<h4>Model Information:</h4>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='debug-label'>Model Type:</span> <span>{str(type(model)) if model else 'N/A'}</span></p>", unsafe_allow_html=True)
    
    classes_info = "N/A"
    if model and hasattr(model, 'names'):
        classes_info = ", ".join([f"{k}: {v}" for k, v in sorted(model.names.items())])
    st.markdown(f"<p><span class='debug-label'>Classes (ID: Name):</span> <span>{classes_info}</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='debug-label'>Number of Classes:</span> <span>{len(model.names) if model and hasattr(model, 'names') else '0'}</span></p>", unsafe_allow_html=True)
    
    st.button("Test Model", key="test_model_btn_expander", help="This will re-check model status and debug info.")


# --- Video Analysis Section ---
st.markdown("<h2><i class='fas fa-video icon'></i> Video Analysis</h2>", unsafe_allow_html=True)
uploaded_video = st.file_uploader(
    "Drag & drop your video here or click to browse",
    type=["mp4", "avi", "mov", "mkv"],
    help="For faster processing, use shorter or lower resolution videos (e.g., under 30 seconds, 720p)."
)

if uploaded_video is not None:
    if model is None:
        st.error("Model is not loaded. Cannot process video.")
    else:
        st.info("Video selected. Click 'Process Video' to start detection.")
        if st.button("Process Video", key="process_video_btn"):
            with st.spinner("Processing video... This may take a while."):
                # Save uploaded video to a temporary file
                tfile_original = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_video.name.split('.')[-1])
                tfile_original.write(uploaded_video.read())
                tfile_original.close()
                original_video_path = tfile_original.name

                original_output_path, detected_output_path, video_props = process_video_file(
                    original_video_path,
                    model,
                    frame_skip_interval,
                    confidence_threshold,
                    max_detections
                )

                if original_output_path and detected_output_path:
                    st.subheader("Detection Results")
                    st.write(f"Total frames: {video_props['total_frames']} | Processed frames: {video_props['processed_frames_count']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("<h4>Original Video</h4>", unsafe_allow_html=True)
                        st.video(original_output_path)
                    with col2:
                        st.markdown("<h4>Detected Objects</h4>", unsafe_allow_html=True)
                        st.video(detected_output_path)
                    
                    # Clean up temporary files after display
                    os.unlink(original_output_path)
                    os.unlink(detected_output_path)
                else:
                    st.error("Video processing failed.")
                
                # Clean up original uploaded temp file
                os.unlink(original_video_path)


# --- Live Webcam Detection Section ---
st.markdown("<h2><i class='fas fa-webcam icon'></i> Live Detection</h2>", unsafe_allow_html=True)

if model is None:
    st.warning("Model not loaded. Live detection is disabled.")
else:
    # Use a unique key for the webrtc_stream_recorder
    webrtc_ctx = webrtc_stream_recorder(
        key="object_detection_webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=lambda: VideoTransformer(model, confidence_threshold, max_detections),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Process frames asynchronously
    )

    if webrtc_ctx.state.playing:
        st.success("Webcam is active. Detecting objects in real-time!")
    else:
        st.info("Click 'Start' to activate webcam for live detection.")

