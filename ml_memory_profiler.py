#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Virtual Memory Footprint Profiler for ML Workloads

This script serves as the main entry point for profiling memory usage
across different ML frameworks (PyTorch, TensorFlow), model sizes,
batch sizes, and hardware configurations.
"""

import os
import sys
import time
import json
import logging
import argparse
import platform
import threading
import datetime
import psutil
from torchvision.models import resnet18
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple

# Import framework-specific modules
try:
    import pytorch_experiments
except ImportError:
    print("PyTorch experiments module not found. PyTorch profiling will be disabled.")

try:
    import tensorflow_experiments
except ImportError:
    print("TensorFlow experiments module not found. TensorFlow profiling will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('ml_profiler')

class SystemInfo:
    """Collects information about the system hardware."""
    
    def __init__(self):
        self.cpu_model = self._get_cpu_model()
        self.cpu_cores = psutil.cpu_count(logical=True)
        self.memory_total = round(psutil.virtual_memory().total / (1024**3), 2)  # GB
        self.platform = platform.platform()
        self.hostname = platform.node()
        
        # GPU information (will be filled later if available)
        self.gpu_available = False
        self.gpu_model = "None"
        self.gpu_memory = 0
        
        # Try to get GPU info if available
        self._detect_gpu()
    
    def _get_cpu_model(self) -> str:
        """Get CPU model information."""
        if platform.system() == "Windows":
            import subprocess
            try:
                output = subprocess.check_output("wmic cpu get name", shell=True).decode().strip()
                lines = output.split('\n')
                if len(lines) >= 2:
                    return lines[1].strip()
            except:
                pass
        
        return platform.processor()
    
    def _detect_gpu(self) -> None:
        """Detect GPU availability and collect information."""
        try:
            # Try import torch first
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_model = torch.cuda.get_device_name(0)
                self.gpu_memory = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)  # GB
                return
        except ImportError:
            pass
        
        try:
            # Try TensorFlow
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                self.gpu_available = True
                self.gpu_model = "TensorFlow GPU"
                # TF doesn't easily expose memory info, so we'll leave it at 0
                return
        except ImportError:
            pass
    
    def __str__(self) -> str:
        """String representation of system information."""
        return (f"CPU: {self.cpu_model} ({self.cpu_cores} cores), "
                f"Memory: {self.memory_total}GB, "
                f"GPU: {'Available - ' + self.gpu_model if self.gpu_available else 'Not available'}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cpu_model': self.cpu_model,
            'cpu_cores': self.cpu_cores,
            'memory_total': self.memory_total,
            'gpu_available': self.gpu_available,
            'gpu_model': self.gpu_model,
            'gpu_memory': self.gpu_memory,
            'platform': self.platform,
            'hostname': self.hostname
        }

class MemoryMonitor:
    """Monitors memory usage of a process and collects data."""
    
    def __init__(self, pid: int, interval: float = 0.1):
        """
        Initialize memory monitor.
        
        Args:
            pid: Process ID to monitor
            interval: Sampling interval in seconds
        """
        self.pid = pid
        self.interval = interval
        self.process = psutil.Process(pid)
        self.running = False
        self.thread = None
        self.data = []
        self.start_time = None
    
    def _collect_memory_stats(self) -> Dict[str, Union[float, int]]:
        """Collect memory statistics for the process."""
        try:
            mem_info = self.process.memory_info()
            
            # Get page fault information
            if hasattr(mem_info, 'num_page_faults'):
                page_faults = mem_info.num_page_faults
                major_faults = getattr(mem_info, 'major_faults', 0)
                minor_faults = page_faults - major_faults
            else:
                # For platforms that don't provide this info
                page_faults = 0
                major_faults = 0
                minor_faults = 0
            
            # Get swap information
            try:
                swap_info = self.process.memory_full_info().swap
            except:
                swap_info = 0
            
            return {
                'timestamp': time.time() - self.start_time,
                'rss': mem_info.rss / (1024**2),  # MB
                'vms': mem_info.vms / (1024**2),  # MB
                'page_faults': page_faults,
                'major_faults': major_faults,
                'minor_faults': minor_faults,
                'swap': swap_info / (1024**2) if swap_info else 0,  # MB
                'cpu_percent': self.process.cpu_percent()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop that collects data at regular intervals."""
        while self.running:
            stats = self._collect_memory_stats()
            if stats:
                self.data.append(stats)
            time.sleep(self.interval)
    
    def start(self) -> None:
        """Start memory monitoring in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Starting memory monitoring for PID {self.pid}")
    
    def stop(self) -> None:
        """Stop memory monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Stopping memory monitoring")
    
    def get_data_frame(self) -> pd.DataFrame:
        """Convert collected data to a pandas DataFrame."""
        return pd.DataFrame(self.data)
    
    def save_data(self, filename: str) -> None:
        """Save collected data to a CSV file."""
        df = self.get_data_frame()
        df.to_csv(filename, index=False)
        logger.info(f"Saving {len(self.data)} memory samples to {filename}")

class Profiler:
    """Main profiler class that coordinates experiments and data collection."""
    
    def __init__(self, 
                 frameworks: List[str],
                 model_sizes: List[str],
                 batch_sizes: List[int],
                 modes: List[str],
                 devices: List[str],
                 output_dir: str = 'results'):
        """
        Initialize profiler.
        
        Args:
            frameworks: List of frameworks to profile (pytorch, tensorflow)
            model_sizes: List of model sizes (small, medium, large)
            batch_sizes: List of batch sizes
            modes: List of modes (train, inference)
            devices: List of devices (cpu, gpu)
            output_dir: Directory to save results
        """
        self.frameworks = frameworks
        self.model_sizes = model_sizes
        self.batch_sizes = batch_sizes
        self.modes = modes
        self.devices = devices
        self.output_dir = output_dir
        self.system_info = SystemInfo()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized profiler with: "
                   f"frameworks={frameworks}, "
                   f"model_sizes={model_sizes}, "
                   f"batch_sizes={batch_sizes}, "
                   f"modes={modes}, "
                   f"devices={devices}")
        logger.info(f"System information: {self.system_info}")
    
    def _get_experiment_name(self, framework: str, model_size: str, 
                            batch_size: int, mode: str, device: str) -> str:
        """Generate a unique experiment name."""
        return f"{framework}_{model_size}_b{batch_size}_{mode}_{device}"
    
    def _get_experiment_dir(self, experiment_name: str) -> str:
        """Get directory path for experiment results."""
        experiment_dir = os.path.join(self.output_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def _save_metadata(self, experiment_dir: str, metadata: Dict[str, Any]) -> None:
        """Save experiment metadata to a JSON file."""
        metadata_file = os.path.join(experiment_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved experiment metadata to {metadata_file}")
    
    def run_experiment(self, framework: str, model_size: str, 
                      batch_size: int, mode: str, device: str) -> None:
        """
        Run a single experiment with the specified parameters.
        
        Args:
            framework: ML framework (pytorch, tensorflow)
            model_size: Model size (small, medium, large)
            batch_size: Batch size
            mode: Mode (train, inference)
            device: Device (cpu, gpu)
        """
        experiment_name = self._get_experiment_name(framework, model_size, batch_size, mode, device)
        experiment_dir = self._get_experiment_dir(experiment_name)
        
        logger.info(f"Running experiment: {experiment_name}")
        
        # Save experiment metadata
        metadata = {
            'framework': framework,
            'model_size': model_size,
            'batch_size': batch_size,
            'mode': mode,
            'device': device,
            'timestamp': datetime.datetime.now().isoformat(),
            'system_info': self.system_info.to_dict()
        }
        self._save_metadata(experiment_dir, metadata)
        
        # Start memory monitoring for current process
        monitor = MemoryMonitor(os.getpid())
        monitor.start()
        
        try:
            # Run framework-specific experiment
            start_time = time.time()
            
            if framework == 'pytorch':
                if 'pytorch_experiments' not in sys.modules:
                    logger.error("PyTorch experiments module not available")
                    return
                pytorch_experiments.run_experiment(
                    model_size=model_size,
                    batch_size=batch_size,
                    mode=mode,
                    device=device
                )
            elif framework == 'tensorflow':
                if 'tensorflow_experiments' not in sys.modules:
                    logger.error("TensorFlow experiments module not available")
                    return
                tensorflow_experiments.run_experiment(
                    model_size=model_size,
                    batch_size=batch_size,
                    mode=mode,
                    device=device
                )
            else:
                logger.error(f"Unsupported framework: {framework}")
                return
            
            elapsed = time.time() - start_time
            logger.info(f"Experiment completed in {elapsed:.2f} seconds")
            
        finally:
            # Stop memory monitoring and save data
            monitor.stop()
            memory_data_file = os.path.join(experiment_dir, "memory_data.csv")
            monitor.save_data(memory_data_file)
            
            logger.info(f"Experiment completed: {experiment_name}")
    
    def run_all_experiments(self) -> None:
        """Run all experiments defined by the configuration."""
        total_experiments = (len(self.frameworks) * len(self.model_sizes) * 
                           len(self.batch_sizes) * len(self.modes) * len(self.devices))
        
        current = 0
        for framework in self.frameworks:
            for model_size in self.model_sizes:
                for batch_size in self.batch_sizes:
                    for mode in self.modes:
                        for device in self.devices:
                            # Skip GPU experiments if GPU is not available
                            if device == 'gpu' and not self.system_info.gpu_available:
                                logger.warning(f"Skipping GPU experiment as no GPU is available: "
                                             f"{framework}_{model_size}_b{batch_size}_{mode}_{device}")
                                continue
                            
                            current += 1
                            logger.info(f"[{current}/{total_experiments}] Running experiment: "
                                       f"{framework}_{model_size}_b{batch_size}_{mode}_{device}")
                            
                            try:
                                self.run_experiment(
                                    framework=framework,
                                    model_size=model_size,
                                    batch_size=batch_size,
                                    mode=mode,
                                    device=device
                                )
                                logger.info(f"Completed experiment: "
                                           f"{framework}_{model_size}_b{batch_size}_{mode}_{device}")
                            except Exception as e:
                                logger.error(f"Error during experiment "
                                           f"{framework}_{model_size}_b{batch_size}_{mode}_{device}: {str(e)}")
    
    def generate_report(self) -> None:
        """Generate summary report from all experiments."""
        # TODO: Implement report generation
        logger.info("Report generation not yet implemented")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Virtual Memory Footprint Profiler for ML Workloads'
    )
    
    parser.add_argument('--frameworks', nargs='+', choices=['pytorch', 'tensorflow'],
                        default=['pytorch', 'tensorflow'],
                        help='ML frameworks to profile')
    
    parser.add_argument('--model-sizes', nargs='+', choices=['small', 'medium', 'large'],
                        default=['small', 'medium', 'large'],
                        help='Model sizes to profile')
    
    parser.add_argument('--batch-sizes', nargs='+', type=int,
                        default=[16, 32, 64],
                        help='Batch sizes to profile')
    
    parser.add_argument('--modes', nargs='+', choices=['train', 'inference'],
                        default=['train', 'inference'],
                        help='Modes to profile')
    
    parser.add_argument('--devices', nargs='+', choices=['cpu', 'gpu'],
                        default=['cpu', 'gpu'],
                        help='Devices to profile')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--single-experiment', action='store_true',
                        help='Run only a single experiment with the first specified parameters')
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the profiler."""
    args = parse_args()
    
    # Initialize file logger
    os.makedirs(args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(args.output_dir, f'profile_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Initialize profiler
    profiler = Profiler(
        frameworks=args.frameworks,
        model_sizes=args.model_sizes,
        batch_sizes=args.batch_sizes,
        modes=args.modes,
        devices=args.devices,
        output_dir=args.output_dir
    )
    
    # Run experiments
    if args.single_experiment:
        # Run just one experiment with the first specified parameters
        profiler.run_experiment(
            framework=args.frameworks[0],
            model_size=args.model_sizes[0],
            batch_size=args.batch_sizes[0],
            mode=args.modes[0],
            device=args.devices[0]
        )
    else:
        # Run all experiments
        profiler.run_all_experiments()
    
    # Generate report
    profiler.generate_report()

if __name__ == "__main__":
    main()