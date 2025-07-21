from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
import json
from ultralytics import YOLO
import tempfile
import uuid
import sys # Added for sys.path manipulation
from trash_classes import get_class_name_short, get_class_color, update_mapping_from_model, EXPECTED_CLASSES # Added EXPECTED_CLASSES

app = Flask(__name__)
CORS(app)

# Ensure the current directory is in sys.path for module imports
# This is crucial for deployment environments where current working directory might not be in path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Global variable to hold the model (and allow reloading)
model = None

# Function to load the model, allowing it to be called multiple times (for reload)
def load_yolo_model():
    global model
    try:
        print("🔄 Loading YOLO model...")
        # Ensure best.pt is in the same directory as app.py or provide full path
        model = YOLO('best.pt') 
        print("✅ Model loaded successfully")
        
        if hasattr(model, 'names') and model.names:
            print("🔍 Model class names:", model.names)
            update_mapping_from_model(model.names)
        else:
            print("⚠️ No class names found in model or model.names is empty")
            # If model has no names, ensure a default mapping or clear it
            update_mapping_from_model({}) # Clear or reset to default if no names
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        return False

# Initial model load when the app starts
model_load_success = load_yolo_model()

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store processed frames for video recreation
# WARNING: This uses server memory. For production, consider external storage or
# more efficient frame handling (e.g., streaming directly or processing chunks).
processed_frames_storage = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    # Check if OpenCV is available
    opencv_available = False
    try:
        import cv2
        opencv_available = True
    except ImportError:
        pass

    # Check if YOLO is available (more deeply, if it can import YOLO class)
    yolo_available = False
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError:
        pass

    model_file_exists = os.path.exists('best.pt')
    model_loaded = model is not None

    model_class_names = {}
    if model_loaded and hasattr(model, 'names'):
        model_class_names = model.names
    
    # Check if trash_classes.py imported correctly
    trash_classes_imported = False
    try:
        from trash_classes import EXPECTED_CLASSES
        if EXPECTED_CLASSES: # Check if it has content
            trash_classes_imported = True
    except ImportError:
        pass


    return jsonify({
        'status': 'healthy', 
        'message': 'Underwater Trash Detection System is running',
        'model_loaded': model_loaded,
        'model_type': str(type(model)) if model_loaded else 'N/A',
        'model_class_names': model_class_names,
        'num_classes': len(model_class_names) if model_loaded else 0,
        'environment_check': {
            'opencv_available': opencv_available,
            'yolo_available': yolo_available,
            'trash_classes_imported': trash_classes_imported,
            'model_file_exists': model_file_exists
        }
    })

@app.route('/reload_model', methods=['POST'])
def reload_model():
    print("Attempting to reload model...")
    success = load_yolo_model()
    if success:
        return jsonify({'status': 'success', 'message': 'Model reloaded successfully!'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to reload model. Check server logs.'}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model is not loaded. Please check if best.pt file exists and is valid.'}), 500
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Check file extension
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
        file_extension = video_file.filename.rsplit('.', 1)[1].lower() if '.' in video_file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({'error': f'Unsupported file format. Please use: {", ".join(allowed_extensions)}'}), 400
        
        # Get processing parameters
        frame_skip = int(request.form.get('frame_skip', 5))
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        max_detections = int(request.form.get('max_detections', 20))
        
        print(f"⚙️ Processing parameters: frame_skip={frame_skip}, confidence_threshold={confidence_threshold}, max_detections={max_detections}")
        
        # Save the uploaded video
        # Using tempfile for safer temporary file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER']) as temp_video_file:
            video_file.save(temp_video_file.name)
            video_path = temp_video_file.name
        
        # Process the video with parameters
        results = process_video(video_path, frame_skip, confidence_threshold, max_detections)
        
        # Clean up the uploaded file
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"❌ Error in upload_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def process_video(video_path, frame_skip=5, confidence_threshold=0.5, max_detections=20):
    """Process video frame by frame and detect trash"""
    # Check if model is loaded
    if model is None:
        raise Exception("Model is not loaded. Please check if best.pt file exists and is valid.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video file. Please check if the file is valid.")
    
    frame_results = []
    frame_count = 0
    processed_frames = []  # Store frames for video recreation
    
    # Get original video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frames based on frame_skip parameter
        if frame_count % frame_skip == 0:
            # Create a copy of the frame for processing
            processed_frame = frame.copy()
            
            # Run inference on the frame
            results = model(frame)
            
            # Process results
            detection_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if we've reached max detections
                        if detection_count >= max_detections:
                            break
                            
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Apply confidence threshold
                        if confidence < confidence_threshold:
                            continue
                        
                        # Debug: Print class ID being detected
                        # print(f"🔍 Detected Class ID: {class_id} (confidence: {confidence:.2f})")
                        
                        # Get class name and color
                        class_name = get_class_name_short(class_id)
                        color = get_class_color(class_id)
                        
                        # Draw bounding box on processed frame
                        cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(processed_frame, label, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        detection_count += 1
            
            # Convert frame to base64 for frontend
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_results.append({
                'frame_number': frame_count,
                'image': frame_base64,
                'detections': detection_count
            })
            
            # Store processed frame for video recreation
            processed_frames.append(processed_frame)
        
        frame_count += 1
    
    cap.release()
    
    # Generate unique session ID for this processing
    session_id = str(uuid.uuid4())
    
    # Store processed frames and video properties
    processed_frames_storage[session_id] = {
        'frames': processed_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': frame_count,
        'processed_frames': len(processed_frames)
    }
    
    return {
        'frames': frame_results,
        'total_frames': frame_count,
        'processed_frames': len(frame_results),
        'session_id': session_id
    }

@app.route('/recreate_video', methods=['POST'])
def recreate_video():
    """Recreate video from processed frames"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in processed_frames_storage:
            return jsonify({'error': 'Invalid session ID or no processed frames found'}), 400
        
        storage_data = processed_frames_storage[session_id]
        frames = storage_data['frames']
        fps = storage_data['fps']
        width = storage_data['width']
        height = storage_data['height']
        
        # Create temporary video file
        # Using tempfile for safer temporary file creation and management
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=app.config['UPLOAD_FOLDER']) as temp_video_output:
            video_path = temp_video_output.name
        
        # Initialize video writer
        # Ensure the codec ('mp4v' or 'XVID') is supported by your OpenCV build
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not open video writer for path: {video_path}")
            return jsonify({'error': 'Could not initialize video writer. Check codecs.'}), 500

        # Write frames to video
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Clean up frames from storage after video recreation
        del processed_frames_storage[session_id]

        # Return video file
        return send_file(video_path, as_attachment=True, download_name=f"detected_trash_video.mp4")
    
    except Exception as e:
        print(f"❌ Error in recreate_video: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame from webcam"""
    try:
        data = request.get_json()
        frame_data = data['frame']
        confidence_threshold = float(data.get('confidence_threshold', 0.5))
        max_detections = int(data.get('max_detections', 20))
        
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model is not loaded. Please check if best.pt file exists and is valid.'}), 500

        # Remove data URL prefix
        frame_data = frame_data.split(',')[1]
        
        # Decode base64 image
        image_data = base64.b64decode(frame_data)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = model(frame)
        
        # Process results
        detections = []
        detection_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if we've reached max detections
                    if detection_count >= max_detections:
                        break
                        
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Apply confidence threshold
                    if confidence < confidence_threshold:
                        continue
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': get_class_name_short(class_id)
                    })
                    
                    detection_count += 1
        
        return jsonify({'detections': detections})
    
    except Exception as e:
        print(f"❌ Error in process_frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # In debug=True, the app will reload on code changes, which is useful for development.
    # For production, set debug=False.
    app.run(debug=True, host='0.0.0.0', port=port) # Changed debug to True for local dev