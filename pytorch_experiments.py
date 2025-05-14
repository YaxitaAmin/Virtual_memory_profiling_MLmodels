"""
PyTorch experiment implementation for memory profiling.

This module provides implementations of different model architectures
and experiment configurations for PyTorch with fixes for multiprocessing issues.
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import resnet18
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Configure logging
logger = logging.getLogger('ml_profiler.pytorch')
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Model Definitions
class SmallModel(nn.Module):
    """LeNet-like small model."""
    def __init__(self, num_classes=10):
        super(SmallModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MediumModel(nn.Module):
    """ResNet-like medium model."""
    def __init__(self, num_classes=10):
        super(MediumModel, self).__init__()
        # Use a pre-defined ResNet but without pre-trained weights
        self.base_model = resnet18(weights=None)
        # Replace the final layer for our number of classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.base_model(x)

class TransformerBlock(nn.Module):
    """Transformer block for the large model."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x

class LargeModel(nn.Module):
    """BERT-like large model using transformers."""
    def __init__(self, max_length=128, vocab_size=30000, embed_dim=768, 
                 num_heads=12, ff_dim=3072, num_transformer_blocks=6, num_classes=10):
        super(LargeModel, self).__init__()
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_transformer_blocks)
        ])
        
        # Classification head
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self.max_length = max_length
        self.embed_dim = embed_dim
        
    def forward(self, x):
        # Create position IDs (0, 1, 2, ..., max_length-1)
        batch_size = x.shape[0]
        positions = torch.arange(0, self.max_length, device=x.device).expand(batch_size, -1)
        
        # Get embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

def get_model(model_size: str, num_classes=10):
    """
    Get model architecture based on size.
    
    Args:
        model_size: small, medium, or large
        num_classes: number of output classes
        
    Returns:
        PyTorch model
    """
    if model_size == 'small':
        return SmallModel(num_classes)
    elif model_size == 'medium':
        return MediumModel(num_classes)
    elif model_size == 'large':
        return LargeModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

# Define dataset outside of any function to make it picklable
class DummyTextDataset(Dataset):
    """Custom dataset for text data."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_cifar10_data():
    """Load and preprocess CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    return trainset, testset

def create_dummy_text_data(num_samples=10000, max_length=128, num_classes=10):
    """Create dummy text data for transformer models."""
    # Random token IDs
    x_data = torch.randint(0, 30000, (num_samples, max_length))
    y_data = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train and test
    train_size = int(0.8 * num_samples)
    x_train, x_test = x_data[:train_size], x_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]
    
    # Create datasets
    trainset = DummyTextDataset(x_train, y_train)
    testset = DummyTextDataset(x_test, y_test)
    
    return trainset, testset

def get_dataset(model_size):
    """Get appropriate dataset based on model size."""
    if model_size == 'large':
        return create_dummy_text_data()
    else:
        return load_cifar10_data()

def run_experiment(model_size: str, batch_size: int, mode: str, device: str) -> None:
    """
    Run a PyTorch experiment with specified parameters.
    
    Args:
        model_size: small, medium, or large
        batch_size: batch size for training/inference
        mode: train or inference
        device: cpu or gpu
    """
    logger.info(f"Starting PyTorch experiment: {model_size} model, batch_size={batch_size}, mode={mode}, device={device}")
    
    # Set device
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Get dataset
    trainset, testset = get_dataset(model_size)
    
    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get model and move to device
    model = get_model(model_size)
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    
    if mode == 'train':
        # Training mode
        model.train()
        
        # Train for 1 epoch
        for i, (inputs, labels) in enumerate(train_loader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            if i % 100 == 0:
                logger.info(f'Batch {i}, Loss: {loss.item():.4f}')
                
            # Limit to a few batches for quick demo
            if i >= 5:
                break
    else:
        # Inference mode
        model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Limit to a few batches for quick demo
                if i >= 5:
                    break
        
        accuracy = 100 * correct / total
        logger.info(f'Inference - Accuracy: {accuracy:.2f}%')
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed PyTorch experiment in {elapsed_time:.2f} seconds")

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
        print("Usage: python pytorch_experiments.py <model_size> <batch_size> <mode> <device>")
        print("Example: python pytorch_experiments.py medium 32 train cpu")