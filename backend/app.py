"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     VideoMotion AI - REST API Server                         ║
║              Activity Recognition using CNN-LSTM Architecture                ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module implements the Flask-based REST API for video activity recognition.
It provides endpoints for video upload, processing, and classification using
a pre-trained CNN-LSTM neural network model.

Author: Deep Learning Assignment
Version: 2.0.0
"""

import os
import json
import logging
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model_loader import initialize_model, fetch_activity_labels
from video_utils import process_video_frames, prepare_model_input

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)-8s │ %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
class AppConfig:
    """Centralized application configuration"""
    UPLOAD_DIRECTORY = 'uploads'
    SUPPORTED_FORMATS = {'mp4', 'avi', 'mov', 'mpg', 'mpeg', 'mkv', 'webm'}
    MAX_FILE_SIZE_MB = 100
    MAX_CONTENT_LENGTH = MAX_FILE_SIZE_MB * 1024 * 1024

app.config['UPLOAD_FOLDER'] = AppConfig.UPLOAD_DIRECTORY
app.config['MAX_CONTENT_LENGTH'] = AppConfig.MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(AppConfig.UPLOAD_DIRECTORY, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("Initializing CNN-LSTM recognition model...")
recognition_model = initialize_model()
activity_labels = fetch_activity_labels()
logger.info(f"Model ready! Recognizable activities: {len(activity_labels)}")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def validate_file_extension(filename: str) -> bool:
    """
    Verify if the uploaded file has a supported video format.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        Boolean indicating if the file format is supported
    """
    if '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in AppConfig.SUPPORTED_FORMATS


def create_response(success: bool, data: dict = None, error: str = None, 
                   message: str = None, status_code: int = 200):
    """
    Generate standardized API response format.
    
    Args:
        success: Operation success status
        data: Response payload data
        error: Error identifier
        message: Human-readable message
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_json, status_code)
    """
    response = {
        'success': success,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if data:
        response.update(data)
    if error:
        response['error'] = error
    if message:
        response['message'] = message
        
    return jsonify(response), status_code


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def health_check():
    """
    System health verification endpoint.
    
    Returns:
        JSON object containing API status and available endpoints.
    """
    return create_response(
        success=True,
        data={
            'status': 'operational',
            'service': 'VideoMotion AI',
            'model_info': {
                'architecture': 'CNN-LSTM Hybrid',
                'backbone': 'MobileNetV2',
                'temporal_module': 'LSTM-64',
                'categories': len(activity_labels)
            },
            'available_endpoints': {
                '/': 'GET - Health check and API information',
                '/categories': 'GET - List all recognizable activities',
                '/predict': 'POST - Analyze video for activity recognition'
            }
        },
        message='VideoMotion AI API is operational'
    )


@app.route('/categories', methods=['GET'])
def list_categories():
    """
    Retrieve available activity categories.
    
    Returns:
        JSON object containing all recognizable activity classes.
    """
    return create_response(
        success=True,
        data={
            'categories': activity_labels,
            'total': len(activity_labels)
        }
    )


# Keep old endpoint for backward compatibility
@app.route('/classes', methods=['GET'])
def get_classes():
    """Legacy endpoint - redirects to /categories"""
    return list_categories()


@app.route('/predict', methods=['POST'])
def analyze_video():
    """
    Process uploaded video and predict activity class.
    
    Expects:
        POST request with video file in 'video' field (multipart/form-data)
    
    Returns:
        JSON object containing:
        - Predicted activity label
        - Confidence score
        - Probability distribution across all classes
    """
    try:
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Validate request contains video file
        # ─────────────────────────────────────────────────────────────────────
        if 'video' not in request.files:
            return create_response(
                success=False,
                error='missing_file',
                message='No video file provided. Please include a video with key "video"',
                status_code=400
            )
        
        uploaded_file = request.files['video']
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Validate file selection
        # ─────────────────────────────────────────────────────────────────────
        if uploaded_file.filename == '':
            return create_response(
                success=False,
                error='empty_filename',
                message='No file selected. Please choose a video file to analyze',
                status_code=400
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Validate file format
        # ─────────────────────────────────────────────────────────────────────
        if not validate_file_extension(uploaded_file.filename):
            supported = ', '.join(sorted(AppConfig.SUPPORTED_FORMATS))
            return create_response(
                success=False,
                error='unsupported_format',
                message=f'Unsupported video format. Accepted formats: {supported}',
                status_code=400
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Save file temporarily for processing
        # ─────────────────────────────────────────────────────────────────────
        safe_filename = secure_filename(uploaded_file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        uploaded_file.save(temp_filepath)
        logger.info(f"Processing video: {safe_filename}")
        
        try:
            # ─────────────────────────────────────────────────────────────────
            # STEP 5: Extract and preprocess video frames
            # ─────────────────────────────────────────────────────────────────
            extracted_frames = process_video_frames(temp_filepath)
            
            if extracted_frames is None:
                return create_response(
                    success=False,
                    error='processing_failed',
                    message='Unable to extract frames from video. The file may be corrupted or use an unsupported codec.',
                    status_code=400
                )
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 6: Prepare input tensor for model
            # ─────────────────────────────────────────────────────────────────
            model_input = prepare_model_input(extracted_frames)
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 7: Execute model inference
            # ─────────────────────────────────────────────────────────────────
            prediction_output = recognition_model.predict(model_input, verbose=0)
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 8: Process prediction results
            # ─────────────────────────────────────────────────────────────────
            predicted_idx = int(prediction_output[0].argmax())
            confidence_score = float(prediction_output[0][predicted_idx])
            predicted_activity = activity_labels[predicted_idx]
            
            # Build probability distribution
            probability_distribution = {
                activity_labels[i]: round(float(prediction_output[0][i]), 6)
                for i in range(len(activity_labels))
            }
            
            # Sort by probability (descending)
            sorted_distribution = dict(
                sorted(probability_distribution.items(), 
                       key=lambda item: item[1], 
                       reverse=True)
            )
            
            logger.info(f"Prediction: {predicted_activity} ({confidence_score*100:.2f}%)")
            
            return create_response(
                success=True,
                data={
                    'prediction': {
                        'action': predicted_activity,
                        'confidence': round(confidence_score * 100, 2),
                        'class_index': predicted_idx
                    },
                    'all_predictions': sorted_distribution
                },
                message=f'Activity detected: {predicted_activity} with {confidence_score*100:.2f}% confidence'
            )
            
        finally:
            # ─────────────────────────────────────────────────────────────────
            # CLEANUP: Remove temporary file
            # ─────────────────────────────────────────────────────────────────
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
                logger.debug(f"Cleaned up temporary file: {safe_filename}")
                
    except Exception as exc:
        logger.error(f"Prediction error: {str(exc)}")
        return create_response(
            success=False,
            error='internal_error',
            message=f'An error occurred during analysis: {str(exc)}',
            status_code=500
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(413)
def handle_file_too_large(error):
    """Handle requests exceeding maximum file size"""
    return create_response(
        success=False,
        error='file_too_large',
        message=f'File exceeds maximum size limit of {AppConfig.MAX_FILE_SIZE_MB}MB',
        status_code=413
    )


@app.errorhandler(500)
def handle_server_error(error):
    """Handle unexpected server errors"""
    logger.error(f"Internal server error: {str(error)}")
    return create_response(
        success=False,
        error='server_error',
        message='An unexpected error occurred. Please try again later.',
        status_code=500
    )


@app.errorhandler(404)
def handle_not_found(error):
    """Handle requests to non-existent endpoints"""
    return create_response(
        success=False,
        error='not_found',
        message='The requested endpoint does not exist. Check /  for available endpoints.',
        status_code=404
    )


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          VideoMotion AI - Activity Recognition API          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Status     : Active                                         ║")
    print(f"║  Endpoint   : http://127.0.0.1:5000                         ║")
    print(f"║  Categories : {len(activity_labels):2d} activity classes                            ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Available Routes:                                           ║")
    print("║    GET  /           - API information                        ║")
    print("║    GET  /categories - List activities                        ║")
    print("║    POST /predict    - Analyze video                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
