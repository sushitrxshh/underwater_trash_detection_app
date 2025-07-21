document.addEventListener('DOMContentLoaded', () => {
    let videoStream = null;
    let webcamInterval = null;
    let isProcessing = false;
    let currentSessionId = null; // Store current session ID for video recreation

    // --- Element References ---
    // Video Analysis
    const uploadArea = document.getElementById('uploadArea');
    const videoInput = document.getElementById('videoInput');
    const chooseVideoBtn = document.getElementById('chooseVideoBtn');
    const processVideoBtn = document.getElementById('processVideoBtn');
    
    // Live Webcam Detection
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopWebcamBtn = document.getElementById('stopWebcamBtn');
    const webcam = document.getElementById('webcam');
    const webcamCanvas = document.getElementById('webcamCanvas');
    const webcamCtx = webcamCanvas.getContext('2d');
    
    // Global Status & Results
    const statusDisplay = document.getElementById('status');
    const resultsSection = document.getElementById('resultsSection');
    const loadingIndicator = document.getElementById('loading');
    const framesGrid = document.getElementById('framesGrid');
    const resultsInfo = document.getElementById('resultsInfo');
    const recreateVideoBtn = document.getElementById('recreateVideoBtn');
    const videoRecreationStatus = document.getElementById('videoRecreationStatus');
    const checkStatusBtn = document.getElementById('checkStatusBtn');
    
    // Slider controls (shared for both video and image processing)
    const frameSkipSlider = document.getElementById('frameSkip');
    const frameSkipValue = document.getElementById('frameSkipValue');
    const confidenceSlider = document.getElementById('confidenceThreshold');
    const confidenceValue = document.getElementById('confidenceValue');
    const maxDetectionsSlider = document.getElementById('maxDetections');
    const maxDetectionsValue = document.getElementById('maxDetectionsValue');
    
    // --- Helper Functions ---
    function showStatus(message, type) {
        statusDisplay.textContent = message;
        statusDisplay.className = `app-status ${type}`;
        statusDisplay.style.display = 'block';
        setTimeout(() => {
            statusDisplay.style.display = 'none';
        }, 5000);
    }
    
    function showVideoRecreationStatus(message, type) {
        videoRecreationStatus.textContent = message;
        videoRecreationStatus.className = `app-status ${type}`;
        videoRecreationStatus.style.display = 'block';
        setTimeout(() => {
            videoRecreationStatus.style.display = 'none';
        }, 5000);
    }

    // --- Initial Load & Model Status Check ---
    async function checkModelStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            if (data.model_loaded) {
                showStatus('✅ Model loaded successfully and server is running!', 'success');
            } else {
                showStatus('⚠️ Model not loaded. Please check server logs and ensure best.pt is present.', 'error');
            }
        } catch (error) {
            showStatus('❌ Cannot connect to server. Please check if the backend is running.', 'error');
        }
    }

    checkModelStatus();
    checkStatusBtn.addEventListener('click', checkModelStatus);

    // --- Slider Updates ---
    frameSkipSlider.addEventListener('input', () => { frameSkipValue.textContent = frameSkipSlider.value; });
    confidenceSlider.addEventListener('input', () => { confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2); });
    maxDetectionsSlider.addEventListener('input', () => { maxDetectionsValue.textContent = maxDetectionsSlider.value; });

    // --- Video Upload & Processing ---
    let selectedVideoFile = null;

    chooseVideoBtn.addEventListener('click', () => { videoInput.click(); });
    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) { handleVideoFile(e.target.files[0]); }
    });
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
    uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('dragover'); });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('video/')) { handleVideoFile(files[0]); }
        else { showStatus('Please drop a video file.', 'error'); }
    });

    function handleVideoFile(file) {
        selectedVideoFile = file;
        showStatus(`Selected video: ${file.name}`, 'info');
        processVideoBtn.style.display = 'inline-block';
        processVideoBtn.disabled = false;
    }

    processVideoBtn.addEventListener('click', async () => {
        if (!selectedVideoFile) { showStatus('Please select a video file first.', 'error'); return; }
        if (isProcessing) return;

        try {
            const healthResponse = await fetch('/health');
            if (!(await healthResponse.json()).model_loaded) { showStatus('❌ Model not loaded on server. Cannot process video.', 'error'); return; }
        } catch (error) { showStatus('❌ Cannot connect to server. Please check if the backend is running.', 'error'); return; }
        
        isProcessing = true;
        processVideoBtn.disabled = true;
        recreateVideoBtn.style.display = 'none';
        framesGrid.innerHTML = '';
        resultsInfo.textContent = '';
        showStatus('Uploading and processing video... This may take a while.', 'info');
        loadingIndicator.style.display = 'block';
        resultsSection.style.display = 'block';
        videoRecreationStatus.style.display = 'none';

        const formData = new FormData();
        formData.append('video', selectedVideoFile);
        formData.append('frame_skip', frameSkipSlider.value);
        formData.append('confidence_threshold', confidenceSlider.value);
        formData.append('max_detections', maxDetectionsSlider.value);

        try {
            const response = await fetch('/upload_video', { method: 'POST', body: formData });
            const result = await response.json();
            if (!response.ok) { throw new Error(result.error || 'Failed to process video.'); }
            displayVideoResults(result);
            showStatus(`Processed ${result.processed_frames} frames successfully!`, 'success');
        } catch (error) {
            console.error('Error processing video:', error);
            showStatus(`Error: ${error.message}`, 'error');
        } finally {
            isProcessing = false;
            processVideoBtn.disabled = false;
            loadingIndicator.style.display = 'none';
        }
    });

    function displayVideoResults(result) {
        resultsInfo.textContent = `Total frames: ${result.total_frames} | Processed frames: ${result.processed_frames}`;
        currentSessionId = result.session_id;
        
        if (result.processed_frames > 0) {
            recreateVideoBtn.style.display = 'inline-block';
            recreateVideoBtn.disabled = false;
        } else {
            recreateVideoBtn.style.display = 'none';
        }

        framesGrid.innerHTML = '';
        result.frames.forEach(frame => {
            const frameCard = document.createElement('div');
            frameCard.className = 'frame-card';
            frameCard.innerHTML = `
                <img src="data:image/jpeg;base64,${frame.image}" alt="Frame ${frame.frame_number}" class="frame-image">
                <div class="frame-info">
                    <h4>Frame ${frame.frame_number}</h4>
                    <span class="detection-count">${frame.detections} trash items detected</span>
                </div>`;
            framesGrid.appendChild(frameCard);
        });
    }

    // --- Video Recreation ---
    recreateVideoBtn.addEventListener('click', async () => {
        if (!currentSessionId) { showVideoRecreationStatus('No processed video available for recreation.', 'error'); return; }
        
        showVideoRecreationStatus('Creating video with detections...', 'info');
        recreateVideoBtn.disabled = true;
        
        try {
            const response = await fetch('/recreate_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'detected_trash_video.mp4';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                showVideoRecreationStatus('Video created successfully! Download started.', 'success');
            } else {
                const error = await response.json();
                throw new Error(error.error || 'Failed to recreate video.');
            }
        } catch (error) {
            console.error('Error recreating video:', error);
            showVideoRecreationStatus(`Error: ${error.message}`, 'error');
        } finally {
            recreateVideoBtn.disabled = false;
        }
    });

    // --- Webcam Functionality ---
    startWebcamBtn.addEventListener('click', async () => {
        try {
            const healthResponse = await fetch('/health');
            if (!(await healthResponse.json()).model_loaded) { showStatus('❌ Model not loaded on server. Cannot start webcam detection.', 'error'); return; }
        } catch (error) { showStatus('❌ Cannot connect to server. Please check if the backend is running.', 'error'); return; }

        try {
            videoStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' } 
            });
            
            webcam.srcObject = videoStream;
            startWebcamBtn.style.display = 'none';
            stopWebcamBtn.style.display = 'inline-block';
            
            webcam.onloadedmetadata = () => {
                webcamCanvas.width = webcam.videoWidth;
                webcamCanvas.height = webcam.videoHeight;
                if (!webcamInterval) { webcamInterval = setInterval(processWebcamFrame, 100); }
            };
            showStatus('Webcam started successfully! Live detection active.', 'success');
        } catch (error) {
            console.error('Error starting webcam:', error);
            showStatus(`Error starting webcam: ${error.message}. Ensure camera permissions are granted.`, 'error');
        }
    });

    stopWebcamBtn.addEventListener('click', () => {
        if (videoStream) { videoStream.getTracks().forEach(track => track.stop()); videoStream = null; }
        if (webcamInterval) { clearInterval(webcamInterval); webcamInterval = null; }
        webcam.srcObject = null;
        startWebcamBtn.style.display = 'inline-block';
        stopWebcamBtn.style.display = 'none';
        webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
        showStatus('Webcam stopped.', 'info');
    });

    async function processWebcamFrame() {
        if (!videoStream || webcam.videoWidth === 0 || isProcessing) return;

        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = webcam.videoWidth;
        tempCanvas.height = webcam.videoHeight;
        tempCtx.drawImage(webcam, 0, 0, webcam.videoWidth, webcam.videoHeight);
        const frameData = tempCanvas.toDataURL('image/jpeg', 0.8);
        
        try {
            isProcessing = true;
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    frame: frameData,
                    confidence_threshold: confidenceSlider.value,
                    max_detections: maxDetectionsSlider.value
                })
            });
            const result = await response.json();
            if (!response.ok) { throw new Error(result.error || 'Failed to process webcam frame.'); }
            if (result.detections) {
                // Pass webcam, canvas and context for drawing
                drawDetectionsOnCanvas(result.detections, webcam, webcamCanvas, webcamCtx);
            }
        } catch (error) {
            console.error('Error processing webcam frame:', error);
            showStatus(`Error in live detection: ${error.message}`, 'error');
            stopWebcam();
        } finally {
            isProcessing = false;
        }
    }

    // Function to draw detections on a given canvas
    function drawDetectionsOnCanvas(detections, sourceElement, targetCanvas, targetCtx) {
        targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
        
        // Determine original dimensions based on source element (video or image)
        const originalWidth = sourceElement.naturalWidth || sourceElement.videoWidth;
        const originalHeight = sourceElement.naturalHeight || sourceElement.videoHeight;

        // Calculate scaling factors for the target canvas
        const scaleX = targetCanvas.width / originalWidth;
        const scaleY = targetCanvas.height / originalHeight;
        
        // Color mapping (can be extended/moved to backend for more control)
        const colorMap = {
            "Mask": "#00FF00", // Green
            "Can": "#00FFFF", // Cyan
            "Cellphone": "#FF00FF", // Magenta
            "Electronics": "#FF4500", // OrangeRed
            "Glass Bottle": "#FFFF00", // Yellow
            "Glove": "#8A2BE2", // BlueViolet
            "Metal": "#C0C0C0", // Silver
            "Misc" :"#FFD700", // Gold for misc
            "Net": "#FFA500", // Orange
            "Plastic Bag": "#00FF7F", // SpringGreen
            "Plastic Bottle": "#DC143C", // Crimson
            "Plastic": "#1E90FF", // DodgerBlue
            "Rod": "#8B4513", // SaddleBrown
            "Sunglasses": "#FF69B4", // HotPink
            "Tyre": "#4682B4" // SteelBlue
        };

        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const confidence = detection.confidence;
            const className = detection.class_name || 'Trash';

            const displayColor = colorMap[className] || '#FF0000'; // Default to Red

            targetCtx.strokeStyle = displayColor;
            targetCtx.lineWidth = 2;
            targetCtx.fillStyle = displayColor;
            targetCtx.font = '12px Arial';

            // Draw bounding box, applying scaling
            // For webcam, canvas is mirrored, so we need to adjust x-coordinates for drawing.
            // For images, the canvas is not mirrored, so direct scaling.
            let displayX1 = x1 * scaleX;
            let displayX2 = x2 * scaleX;

            if (targetCanvas.id === 'webcamCanvas') { // Apply mirroring correction for webcam
                displayX1 = targetCanvas.width - (x2 * scaleX);
                displayX2 = targetCanvas.width - (x1 * scaleX);
            }
            
            targetCtx.strokeRect(displayX1, y1 * scaleY, displayX2 - displayX1, (y2 - y1) * scaleY);
            
            // Draw label
            const label = `${className} ${(confidence * 100).toFixed(1)}%`;
            let textX = displayX1;
            const textY = y1 * scaleY > 15 ? y1 * scaleY - 5 : y1 * scaleY + 15;
            targetCtx.fillText(label, textX, textY);
        });
    }

    // Initialize slider value displays
    frameSkipValue.textContent = frameSkipSlider.value;
    confidenceValue.textContent = parseFloat(confidenceSlider.value).toFixed(2);
    maxDetectionsValue.textContent = maxDetectionsSlider.value;
});