#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TensorFlow experiment implementation for memory profiling.

This module provides implementations of different model architectures
and experiment configurations for TensorFlow.
"""

import os
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics, applications
from tensorflow.keras.datasets import cifar10
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tensorflow.keras import mixed_precision
import sys
from torchvision.models import resnet18
# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
# Configure logging
logger = logging.getLogger('ml_profiler.tensorflow')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Make TensorFlow logs less verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def clear_session_and_garbage_collect():
    import gc
    import tensorflow as tf
    tf.keras.backend.clear_session()
    gc.collect()
    
# Model Definitions
def create_small_model(input_shape=(32, 32, 3), num_classes=10):
    """Create a LeNet-like small model."""
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), padding='valid', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(num_classes)
    ])
    return model

def create_medium_model(input_shape=(32, 32, 3), num_classes=10):
    """Create a ResNet-like medium model."""
    # For simplicity, we'll use a pre-trained model but with a custom final layer
    base_model = applications.ResNet50(
        weights=None,  # No pre-trained weights to keep memory footprint accurate
        include_top=False,
        input_shape=input_shape
    )
    
    # Add custom classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes)(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def create_transformer_block(embed_dim, num_heads, ff_dim, rate=0.1):
    """Create a transformer block."""
    inputs = layers.Input(shape=(None, embed_dim))
    
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim
    )(inputs, inputs)
    attention_output = layers.Dropout(rate)(attention_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)

    outputs = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    return models.Model(inputs=inputs, outputs=outputs)

def create_large_model(max_length=128, vocab_size=30000, embed_dim=768, num_heads=12, ff_dim=3072, num_transformer_blocks=6, num_classes=10):
    """Create a BERT-like large model using transformers."""
    inputs = layers.Input(shape=(max_length,))
    
    # Token embeddings
    token_embeddings = layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim
    )(inputs)
    
    # Position embeddings
    position_embeddings = layers.Embedding(
        input_dim=max_length, output_dim=embed_dim
    )(tf.range(start=0, limit=max_length, delta=1))
    
    # Explicitly cast both tensors to the same dtype before adding
    dtype = token_embeddings.dtype
    position_embeddings = tf.cast(position_embeddings, dtype)
    
    # Add position embeddings to token embeddings
    embeddings = token_embeddings + position_embeddings
    embeddings = layers.LayerNormalization(epsilon=1e-6)(embeddings)
    embeddings = layers.Dropout(0.1)(embeddings)
    
    # Transformer blocks
    x = embeddings
    for _ in range(num_transformer_blocks):
        x = create_transformer_block(embed_dim, num_heads, ff_dim)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def get_model(model_size: str, input_shape=(32, 32, 3), num_classes=10):
    """
    Get model architecture based on size.
    
    Args:
        model_size: small, medium, or large
        input_shape: shape of input data
        num_classes: number of output classes
        
    Returns:
        TensorFlow model
    """
    if model_size == 'small':
        return create_small_model(input_shape, num_classes)
    elif model_size == 'medium':
        return create_medium_model(input_shape, num_classes)
    elif model_size == 'large':
        # For large model, we'll use our BERT-like transformer
        # This requires different input shape, so we'll handle it in run_experiment
        return create_large_model(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

def load_cifar10_data():
    """Load and preprocess CIFAR-10 dataset."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_dummy_text_data(num_samples=10000, max_length=128, num_classes=10):
    """Create dummy text data for transformer models."""
    # For simplicity, create random token IDs
    x_data = np.random.randint(0, 30000, size=(num_samples, max_length))
    y_data = tf.keras.utils.to_categorical(
        np.random.randint(0, num_classes, size=(num_samples, 1)), 
        num_classes=num_classes
    )
    
    # Split into train and test sets
    train_size = int(0.8 * num_samples)
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    
    return (x_train, y_train), (x_test, y_test)

def run_experiment(model_size: str, batch_size: int, mode: str, device: str) -> None:
    """
    Run a TensorFlow experiment with specified parameters.
    
    Args:
        model_size: small, medium, or large
        batch_size: batch size for training/inference
        mode: train or inference
        device: cpu or gpu
    """
    logger.info(f"Starting TensorFlow experiment: {model_size} model, batch_size={batch_size}, mode={mode}, device={device}")
    
    # Configure device strategy
    if device == 'gpu' and tf.config.list_physical_devices('GPU'):
        with tf.device('/device:GPU:0'):
            _run_experiment_impl(model_size, batch_size, mode)
    else:
        with tf.device('/device:CPU:0'):
            _run_experiment_impl(model_size, batch_size, mode)

def _run_experiment_impl(model_size: str, batch_size: int, mode: str) -> None:
    """Implementation of experiment execution."""
    start_time = time.time()
    
    # Load appropriate data based on model size
    if model_size == 'large':
        # For large models we'll use text-like data
        (x_train, y_train), (x_test, y_test) = create_dummy_text_data()
    else:
        # For small and medium we'll use image data
        (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    
    # Get the model
    if model_size == 'large':
        model = create_large_model()
    else:
        model = get_model(model_size)
        
    # Compile the model with appropriate loss and optimizer
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    if mode == 'train':
        # Training mode
        # Use a smaller subset of data for quick demo
        num_samples = min(10000, len(x_train))
        x_train_subset = x_train[:num_samples]
        y_train_subset = y_train[:num_samples]
        
        # Train for 1 epoch
        history = model.fit(
            x_train_subset, y_train_subset,
            batch_size=batch_size,
            epochs=1,
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Log results
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0]
        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        
        logger.info(f"Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%")
    else:
        # Inference mode
        results = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        logger.info(f"Inference - Val Loss: {results[0]:.4f}, Val Accuracy: {results[1]*100:.2f}%")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed TensorFlow experiment in {elapsed_time:.2f} seconds")
    
    # Clean up to free memory
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Example standalone usage
    if len(sys.argv) >= 5:
        model_size = sys.argv[1]
        batch_size = int(sys.argv[2])
        mode = sys.argv[3]
        device = sys.argv[4]
        run_experiment(model_size, batch_size, mode, device)
    else:
        print("Usage: python tensorflow_experiments.py <model_size> <batch_size> <mode> <device>")
        print("Example: python tensorflow_experiments.py medium 32 train cpu")