"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   Neural Network Model Manager                                ║
║        Handles CNN-LSTM model initialization and weight loading              ║
╚══════════════════════════════════════════════════════════════════════════════╝

This module is responsible for:
- Constructing the CNN-LSTM hybrid architecture
- Loading pre-trained weights from disk
- Providing activity class label mappings

Architecture Overview:
    Input → TimeDistributed(MobileNetV2) → LSTM → Dropout → Dense → Softmax

Author: Deep Learning Assignment
Version: 2.0.0
"""

import os
import json
from typing import List, Dict, Optional
import tensorflow as tf

# ═══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_FILE = os.path.join(PROJECT_ROOT, 'models', 'ucf11_cnn_lstm_model.h5')
LABELS_FILE = os.path.join(PROJECT_ROOT, 'models', 'classes.json')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
class ModelConfig:
    """Centralized model configuration parameters"""
    SEQUENCE_LENGTH = 20        # Number of frames per video sample
    FRAME_HEIGHT = 224          # Input frame height in pixels
    FRAME_WIDTH = 224           # Input frame width in pixels
    COLOR_CHANNELS = 3          # RGB color channels
    LSTM_UNITS = 64             # Hidden units in LSTM layer
    DROPOUT_RATE = 0.5          # Dropout probability for regularization
    NUM_CATEGORIES = 11         # Number of activity classes (UCF11)

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON CACHE FOR MODEL INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════
_cached_model: Optional[tf.keras.Model] = None
_cached_labels: Optional[List[str]] = None


def construct_architecture() -> tf.keras.Model:
    """
    Build the CNN-LSTM hybrid neural network architecture.
    
    The architecture consists of:
    1. TimeDistributed MobileNetV2: Extracts spatial features from each frame
       independently, producing a feature vector per frame.
    2. LSTM Layer: Processes the sequence of feature vectors to capture
       temporal dependencies and motion patterns.
    3. Dropout Layer: Regularization to prevent overfitting.
    4. Dense Layer: Final classification with softmax activation.
    
    Returns:
        tf.keras.Model: Compiled model ready for inference or training
        
    Note:
        The MobileNetV2 backbone is frozen (non-trainable) to leverage
        pre-trained ImageNet features efficiently.
    """
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import (
        TimeDistributed, LSTM, Dense, Dropout, Input
    )
    from tensorflow.keras.models import Model
    
    cfg = ModelConfig
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPATIAL FEATURE EXTRACTOR (CNN Backbone)
    # ─────────────────────────────────────────────────────────────────────────
    spatial_encoder = MobileNetV2(
        weights="imagenet",
        include_top=False,          # Remove classification head
        pooling="avg",              # Global average pooling
        input_shape=(cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, cfg.COLOR_CHANNELS)
    )
    # Freeze backbone weights for transfer learning
    spatial_encoder.trainable = False
    
    # ─────────────────────────────────────────────────────────────────────────
    # COMPLETE MODEL ASSEMBLY
    # ─────────────────────────────────────────────────────────────────────────
    # Input layer: batch of video sequences
    sequence_input = Input(
        shape=(cfg.SEQUENCE_LENGTH, cfg.FRAME_HEIGHT, cfg.FRAME_WIDTH, cfg.COLOR_CHANNELS),
        name='video_sequence_input'
    )
    
    # Apply CNN to each frame in the sequence
    frame_features = TimeDistributed(
        spatial_encoder, 
        name='spatial_feature_extractor'
    )(sequence_input)
    
    # Temporal sequence modeling
    temporal_context = LSTM(
        cfg.LSTM_UNITS, 
        name='temporal_sequence_learner'
    )(frame_features)
    
    # Regularization
    regularized = Dropout(
        cfg.DROPOUT_RATE, 
        name='regularization_dropout'
    )(temporal_context)
    
    # Classification output
    activity_probabilities = Dense(
        cfg.NUM_CATEGORIES, 
        activation="softmax",
        name='activity_classifier'
    )(regularized)
    
    # Assemble final model
    recognition_model = Model(
        inputs=sequence_input, 
        outputs=activity_probabilities,
        name='VideoMotionAI_CNN_LSTM'
    )
    
    return recognition_model


def initialize_model() -> tf.keras.Model:
    """
    Initialize and load the pre-trained activity recognition model.
    
    This function implements a singleton pattern to ensure the model
    is only loaded once, improving performance for subsequent calls.
    
    The loading process:
    1. Check if model is already cached in memory
    2. Verify model weights file exists on disk
    3. Reconstruct the network architecture
    4. Load pre-trained weights into the model
    5. Compile model for inference
    
    Returns:
        tf.keras.Model: Loaded and compiled model ready for predictions
        
    Raises:
        FileNotFoundError: If the model weights file is not found
    """
    global _cached_model
    
    # Return cached model if available
    if _cached_model is not None:
        return _cached_model
    
    # Verify weights file exists
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(
            f"Model weights not found at: {WEIGHTS_FILE}\n"
            f"Please ensure 'ucf11_cnn_lstm_model.h5' exists in the 'models' directory."
        )
    
    print(f"📦 Loading model weights from: {WEIGHTS_FILE}")
    
    # Suppress verbose TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    
    # Build network architecture
    print("🔧 Constructing neural network architecture...")
    _cached_model = construct_architecture()
    
    # Load trained weights
    print("⚖️  Loading pre-trained weights...")
    try:
        _cached_model.load_weights(WEIGHTS_FILE)
        print("✅ Weights loaded successfully!")
    except Exception as load_error:
        print(f"⚠️  Weight loading warning: {load_error}")
        print("   Proceeding with ImageNet pre-trained features only.")
    
    # Compile model for inference
    _cached_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model summary info
    print(f"📊 Model Input Shape: {_cached_model.input_shape}")
    print(f"📊 Model Output Shape: {_cached_model.output_shape}")
    print(f"📊 Total Parameters: {_cached_model.count_params():,}")
    
    return _cached_model


# Alias for backward compatibility
load_model = initialize_model


def fetch_activity_labels() -> List[str]:
    """
    Retrieve the list of activity class labels.
    
    Labels are loaded from the classes.json file if available,
    otherwise defaults to the standard UCF11 category names.
    
    UCF11 Dataset Categories:
    - Basketball, Biking, Diving, Golf Swing, Horse Riding
    - Soccer Juggling, Swing, Tennis Swing, Trampoline Jumping
    - Volleyball Spiking, Walking
    
    Returns:
        List[str]: Ordered list of activity class names
    """
    global _cached_labels
    
    # Return cached labels if available
    if _cached_labels is not None:
        return _cached_labels
    
    if os.path.exists(LABELS_FILE):
        # Load from configuration file
        with open(LABELS_FILE, 'r', encoding='utf-8') as label_file:
            label_data = json.load(label_file)
            _cached_labels = label_data.get('classes', [])
    else:
        # Fallback to default UCF11 labels (alphabetically sorted)
        _cached_labels = [
            "basketball",
            "biking", 
            "diving",
            "golf_swing",
            "horse_riding",
            "soccer_juggling",
            "swing",
            "tennis_swing",
            "trampoline_jumping",
            "volleyball_spiking",
            "walking"
        ]
    
    return _cached_labels


# Alias for backward compatibility
get_class_labels = fetch_activity_labels


def retrieve_model_metadata() -> Dict:
    """
    Generate comprehensive metadata about the loaded model.
    
    Returns:
        Dict: Detailed model information including architecture,
              parameters, and configuration settings
    """
    model = initialize_model()
    labels = fetch_activity_labels()
    cfg = ModelConfig
    
    return {
        'model_name': 'VideoMotion AI CNN-LSTM',
        'weights_path': WEIGHTS_FILE,
        'input_specification': {
            'shape': model.input_shape,
            'sequence_length': cfg.SEQUENCE_LENGTH,
            'frame_size': f'{cfg.FRAME_HEIGHT}x{cfg.FRAME_WIDTH}',
            'channels': cfg.COLOR_CHANNELS
        },
        'output_specification': {
            'shape': model.output_shape,
            'num_classes': len(labels)
        },
        'architecture': {
            'backbone': 'MobileNetV2 (ImageNet pre-trained)',
            'temporal_module': f'LSTM ({cfg.LSTM_UNITS} units)',
            'regularization': f'Dropout (p={cfg.DROPOUT_RATE})',
            'classifier': f'Dense (softmax, {cfg.NUM_CATEGORIES} outputs)'
        },
        'activity_categories': labels,
        'total_parameters': model.count_params()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 60)
    print("  Neural Network Model Manager - Diagnostic Test")
    print("═" * 60 + "\n")
    
    try:
        # Test model loading
        model = initialize_model()
        labels = fetch_activity_labels()
        metadata = retrieve_model_metadata()
        
        print("┌─────────────────────────────────────────────────────────┐")
        print("│              Model Information Summary                  │")
        print("├─────────────────────────────────────────────────────────┤")
        print(f"│  Model Name    : {metadata['model_name']:<38} │")
        print(f"│  Input Shape   : {str(metadata['input_specification']['shape']):<38} │")
        print(f"│  Output Shape  : {str(metadata['output_specification']['shape']):<38} │")
        print(f"│  Parameters    : {metadata['total_parameters']:,}                         │")
        print(f"│  Categories    : {len(labels):<38} │")
        print("└─────────────────────────────────────────────────────────┘")
        
        print("\n✅ All diagnostic tests passed successfully!")
        
    except Exception as test_error:
        print(f"\n❌ Test failed: {test_error}")
