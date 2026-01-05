"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      Video Processing Pipeline                               ║
║           Frame extraction and preprocessing for CNN-LSTM input              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module handles all video-related operations required for activity recognition:
- Reading video files in various formats
- Extracting frames at specified intervals
- Resizing and normalizing frames for neural network input
- Preparing batched tensors for model inference

Processing Pipeline:
    Video File → Frame Extraction → Resize → Normalize → Batch → Model Input

Author: Deep Learning Assignment
Version: 2.0.0
"""

import numpy as np
import cv2
import imageio.v2 as imageio
from typing import Optional, List, Dict, Tuple
import logging

# Configure module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
class VideoConfig:
    """Centralized video preprocessing parameters"""
    FRAMES_PER_VIDEO = 20           # Target number of frames to extract
    TARGET_HEIGHT = 224             # Output frame height (pixels)
    TARGET_WIDTH = 224              # Output frame width (pixels)  
    COLOR_CHANNELS = 3              # RGB color space
    PIXEL_NORMALIZATION = 255.0     # Normalization divisor (maps to 0-1)


def process_video_frames(
    video_filepath: str,
    frame_count: int = VideoConfig.FRAMES_PER_VIDEO,
    target_size: int = VideoConfig.TARGET_HEIGHT
) -> Optional[np.ndarray]:
    """
    Extract and preprocess frames from a video file.
    
    This function performs the following operations:
    1. Opens the video file using imageio (with FFmpeg backend for broad format support)
    2. Extracts up to `frame_count` frames from the video
    3. Resizes each frame to `target_size` x `target_size` pixels
    4. Normalizes pixel values from [0, 255] to [0, 1] range
    5. Pads with zero frames if the video has fewer than required frames
    
    Args:
        video_filepath: Absolute path to the input video file
        frame_count: Number of frames to extract (default: 20)
        target_size: Square dimension for output frames (default: 224)
    
    Returns:
        np.ndarray: Preprocessed frames with shape (frame_count, target_size, target_size, 3)
        None: If frame extraction fails
        
    Example:
        >>> frames = process_video_frames('/path/to/video.mp4')
        >>> print(frames.shape)
        (20, 224, 224, 3)
    """
    extracted_frames: List[np.ndarray] = []
    
    try:
        # ─────────────────────────────────────────────────────────────────────
        # Primary extraction using imageio (FFmpeg backend)
        # ─────────────────────────────────────────────────────────────────────
        video_reader = imageio.get_reader(video_filepath, format='ffmpeg')
        
        # Attempt to retrieve video metadata for logging
        try:
            metadata = video_reader.get_meta_data()
            frame_total = metadata.get('nframes', 'unknown')
            video_fps = metadata.get('fps', 'unknown')
            video_duration = metadata.get('duration', 0)
            logger.info(
                f"Video properties: {frame_total} frames | "
                f"{video_fps} fps | {video_duration:.2f}s duration"
            )
        except Exception:
            pass  # Metadata extraction is optional
        
        # Extract frames from video stream
        for frame_idx, raw_frame in enumerate(video_reader):
            if frame_idx >= frame_count:
                break
            
            # Spatial transformation: resize to target dimensions
            resized_frame = cv2.resize(
                raw_frame, 
                (target_size, target_size),
                interpolation=cv2.INTER_AREA
            )
            
            # Intensity normalization: scale to [0, 1] range
            normalized_frame = resized_frame / VideoConfig.PIXEL_NORMALIZATION
            
            extracted_frames.append(normalized_frame)
        
        video_reader.close()
        
    except Exception as extraction_error:
        logger.warning(f"Primary extraction failed: {extraction_error}")
        
        # ─────────────────────────────────────────────────────────────────────
        # Fallback extraction using OpenCV
        # ─────────────────────────────────────────────────────────────────────
        try:
            extracted_frames = _opencv_frame_extraction(
                video_filepath, frame_count, target_size
            )
            if extracted_frames is None or len(extracted_frames) == 0:
                return None
        except Exception as fallback_error:
            logger.error(f"Fallback extraction also failed: {fallback_error}")
            return None
    
    if len(extracted_frames) == 0:
        logger.error("No frames could be extracted from the video")
        return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Zero-padding for videos shorter than required frame count
    # ─────────────────────────────────────────────────────────────────────────
    while len(extracted_frames) < frame_count:
        zero_frame = np.zeros(
            (target_size, target_size, VideoConfig.COLOR_CHANNELS),
            dtype=np.float32
        )
        extracted_frames.append(zero_frame)
    
    logger.info(f"Successfully extracted {len(extracted_frames)} frames")
    return np.array(extracted_frames, dtype=np.float32)


# Alias for backward compatibility
extract_frames = process_video_frames


def _opencv_frame_extraction(
    video_filepath: str,
    frame_count: int = VideoConfig.FRAMES_PER_VIDEO,
    target_size: int = VideoConfig.TARGET_HEIGHT
) -> Optional[List[np.ndarray]]:
    """
    Fallback frame extraction using OpenCV VideoCapture.
    
    This method is used when the primary imageio extraction fails,
    typically due to codec compatibility issues.
    
    Args:
        video_filepath: Path to the video file
        frame_count: Target number of frames
        target_size: Output frame dimensions
    
    Returns:
        List of preprocessed frame arrays, or None on failure
    """
    frames: List[np.ndarray] = []
    video_capture = cv2.VideoCapture(video_filepath)
    
    if not video_capture.isOpened():
        logger.error(f"OpenCV could not open video: {video_filepath}")
        return None
    
    frames_read = 0
    
    while frames_read < frame_count:
        success, raw_frame = video_capture.read()
        
        if not success:
            break
        
        # Convert BGR (OpenCV default) to RGB color space
        rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to target dimensions
        resized_frame = cv2.resize(
            rgb_frame, 
            (target_size, target_size),
            interpolation=cv2.INTER_AREA
        )
        
        # Normalize pixel intensities
        normalized_frame = resized_frame / VideoConfig.PIXEL_NORMALIZATION
        
        frames.append(normalized_frame)
        frames_read += 1
    
    video_capture.release()
    return frames


def prepare_model_input(frame_sequence: np.ndarray) -> np.ndarray:
    """
    Transform extracted frames into model-compatible input tensor.
    
    The CNN-LSTM model expects input with shape:
        (batch_size, sequence_length, height, width, channels)
    
    This function adds the batch dimension if not present.
    
    Args:
        frame_sequence: Preprocessed frames of shape (seq_len, H, W, C)
    
    Returns:
        np.ndarray: Batched tensor of shape (1, seq_len, H, W, C)
        
    Raises:
        ValueError: If input frames are None or have invalid dimensions
    """
    if frame_sequence is None:
        raise ValueError("Frame sequence cannot be None")
    
    # Validate and adjust tensor dimensions
    if len(frame_sequence.shape) == 4:
        # Add batch dimension: (S, H, W, C) -> (1, S, H, W, C)
        model_input = np.expand_dims(frame_sequence, axis=0)
    elif len(frame_sequence.shape) == 5:
        # Already batched, use as-is
        model_input = frame_sequence
    else:
        raise ValueError(
            f"Invalid frame sequence shape: {frame_sequence.shape}. "
            f"Expected 4D (S, H, W, C) or 5D (B, S, H, W, C) array."
        )
    
    # Ensure float32 dtype for TensorFlow compatibility
    model_input = model_input.astype(np.float32)
    
    logger.info(f"Model input tensor shape: {model_input.shape}")
    return model_input


# Alias for backward compatibility
preprocess_video = prepare_model_input


def validate_video_file(video_filepath: str) -> Dict:
    """
    Perform comprehensive validation on a video file.
    
    Checks performed:
    - File existence on disk
    - File readability with video decoder
    - Presence of extractable frames
    
    Args:
        video_filepath: Path to the video file to validate
    
    Returns:
        Dict containing validation results:
        - valid: Boolean indicating overall validity
        - exists: File exists on disk
        - readable: Can be opened by video decoder
        - has_frames: Contains at least one extractable frame
        - error: Error message if validation failed
    """
    import os
    
    validation_result = {
        'valid': False,
        'filepath': video_filepath,
        'exists': False,
        'readable': False,
        'has_frames': False,
        'error': None
    }
    
    # Check file existence
    if not os.path.exists(video_filepath):
        validation_result['error'] = 'File does not exist on disk'
        return validation_result
    validation_result['exists'] = True
    
    # Attempt to read video
    try:
        video_reader = imageio.get_reader(video_filepath, format='ffmpeg')
        validation_result['readable'] = True
        
        # Verify frame extraction
        frame_counter = 0
        for _ in video_reader:
            frame_counter += 1
            if frame_counter >= 1:
                break
        video_reader.close()
        
        if frame_counter > 0:
            validation_result['has_frames'] = True
            validation_result['valid'] = True
        else:
            validation_result['error'] = 'Video contains no extractable frames'
            
    except Exception as validation_error:
        validation_result['error'] = str(validation_error)
    
    return validation_result


def extract_video_metadata(video_filepath: str) -> Dict:
    """
    Extract detailed metadata from a video file.
    
    Args:
        video_filepath: Path to the video file
    
    Returns:
        Dict containing video properties:
        - path: File path
        - fps: Frames per second
        - duration: Video duration in seconds
        - total_frames: Total frame count
        - resolution: Video dimensions (width, height)
        - error: Error message if extraction failed
    """
    metadata = {
        'path': video_filepath,
        'fps': None,
        'duration': None,
        'total_frames': None,
        'resolution': None
    }
    
    try:
        video_reader = imageio.get_reader(video_filepath, format='ffmpeg')
        raw_metadata = video_reader.get_meta_data()
        
        metadata['fps'] = raw_metadata.get('fps')
        metadata['duration'] = raw_metadata.get('duration')
        metadata['total_frames'] = raw_metadata.get('nframes')
        metadata['resolution'] = raw_metadata.get('size')
        
        video_reader.close()
    except Exception as metadata_error:
        metadata['error'] = str(metadata_error)
    
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    
    print("\n" + "═" * 60)
    print("  Video Processing Pipeline - Diagnostic Tool")
    print("═" * 60 + "\n")
    
    if len(sys.argv) > 1:
        test_video_path = sys.argv[1]
        print(f"📁 Testing video: {test_video_path}\n")
        
        # Run validation
        print("┌─ Validation Results ─────────────────────────────────────┐")
        validation = validate_video_file(test_video_path)
        for key, value in validation.items():
            print(f"│  {key:<15}: {str(value):<40} │")
        print("└──────────────────────────────────────────────────────────┘\n")
        
        if validation['valid']:
            # Extract and display metadata
            print("┌─ Video Metadata ─────────────────────────────────────────┐")
            metadata = extract_video_metadata(test_video_path)
            for key, value in metadata.items():
                if value is not None:
                    print(f"│  {key:<15}: {str(value):<40} │")
            print("└──────────────────────────────────────────────────────────┘\n")
            
            # Test frame extraction
            print("🎬 Extracting frames...")
            frames = process_video_frames(test_video_path)
            
            if frames is not None:
                print(f"✅ Extracted shape: {frames.shape}")
                
                # Test model input preparation
                model_input = prepare_model_input(frames)
                print(f"✅ Model input shape: {model_input.shape}")
                print("\n✅ All tests passed successfully!")
            else:
                print("❌ Frame extraction failed")
    else:
        print("Usage: python video_utils.py <path_to_video>\n")
        print("Configuration Summary:")
        print(f"  ├─ Frames per video  : {VideoConfig.FRAMES_PER_VIDEO}")
        print(f"  ├─ Frame dimensions  : {VideoConfig.TARGET_HEIGHT}×{VideoConfig.TARGET_WIDTH}")
        print(f"  ├─ Color channels    : {VideoConfig.COLOR_CHANNELS} (RGB)")
        print(f"  └─ Pixel range       : [0, 1] (normalized)")
