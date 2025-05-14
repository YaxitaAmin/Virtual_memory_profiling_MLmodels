# Virtual Memory Profiling for ML Models

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://virtualmemoryprofilingmlmodels.streamlit.app/)

A comprehensive tool for analyzing and profiling memory usage patterns across popular machine learning frameworks (TensorFlow and PyTorch). This project helps data scientists and machine learning engineers optimize memory utilization in their ML workflows.

## 🌟 Features

- **Cross-Framework Memory Profiling**: Compare memory usage between TensorFlow and PyTorch implementations
- **Detailed Metrics**: Track virtual memory allocation, peak usage, and footprint over time
- **Interactive Visualization**: Visualize memory patterns through an intuitive Streamlit dashboard
- **Model Comparison**: Compare memory efficiency across different model architectures and batch sizes
- **Training Phase Analysis**: Identify memory-intensive phases during model training

## 📊 Dashboard

The project includes a [live Streamlit dashboard](https://virtualmemoryprofilingmlmodels.streamlit.app/) that visualizes memory profiling results and allows for interactive exploration of the data.

![Screenshot 2025-05-13 171015](https://github.com/user-attachments/assets/b881022b-47fc-4c59-a14d-710a90f287de)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YaxitaAmin/Virtual_memory_profiling_MLmodels.git
cd Virtual_memory_profiling_MLmodels

# Install dependencies
pip install -r requirements.txt
```

## 📋 Requirements

See `requirements.txt` for a complete list of dependencies. Main requirements include:

- Python 3.8+
- TensorFlow 2.x
- PyTorch 1.x+
- Streamlit
- Pandas
- Matplotlib
- memory_profiler

## 🔍 Project Methodology

### Experimental Setup
- **Frameworks:** PyTorch and TensorFlow
- **Model Sizes:** Small, Medium, Large
- **Batch Sizes:** 16, 32, 64
- **Execution Modes:** Training and Inference
- **Hardware:** CPU and GPU

### Metrics Collected
- Mean Memory Usage (MB)
- Maximum Memory Usage (MB)
- Minimum Memory Usage (MB)
- Standard Deviation
- Median Memory Usage

### Tools Used
- Custom memory profiler (`ml_memory_profiler.py`)
- Streamlit dashboard for visualization and filtering

## 📁 Project Structure

```
.
├── .devcontainer/         # Development container configuration
├── data/                  # Input datasets
├── results/               # CSV output files with profiling results
├── .gitattributes         # Git attributes
├── .gitignore             # Git ignore rules
├── __init__.py            # Package initialization
├── ml_memory_profiler.py  # Main memory profiling implementation
├── pytorch_experiments.py # PyTorch-specific experiments
├── requirements.txt       # Project dependencies
├── streamlit-app.py       # Streamlit dashboard application
└── tensorflow_experiments.py # TensorFlow-specific experiments
```

## 📈 Key Findings & Results

Based on comprehensive analysis of memory usage across frameworks:

### Framework Comparison
- **PyTorch Average Memory Usage:** 1.53 MB
- **TensorFlow Average Memory Usage:** 3.83 MB
- **Ratio (TensorFlow/PyTorch):** 2.51x

### Batch Size Impact
**PyTorch (Large Model, Inference, CPU):**
- Batch 16: 1.86 MB
- Batch 32: 3.20 MB
- Batch 64: 6.24 MB
- **Growth Factor (16→64):** 3.35x

**TensorFlow (Large Model, Inference, CPU):**
- Batch 16: 6.25 MB
- Batch 32: 6.25 MB
- Batch 64: 6.25 MB
- **Growth Factor (16→64):** 1.00x

### CPU vs GPU Memory Usage
**Large Model, Batch Size 64, Inference Mode:**

**PyTorch:**
- CPU: 6.24 MB
- GPU: 1.49 MB
- **CPU/GPU Ratio:** 4.20x

**TensorFlow:**
- CPU: 6.25 MB
- GPU: 1.42 MB
- **CPU/GPU Ratio:** 4.40x

### Training vs Inference Memory Usage
**PyTorch (Large Model, Batch 16, CPU):**
- Inference: 1.86 MB
- Training: 2.27 MB
- **Training/Inference Ratio:** 1.22x

**TensorFlow (Large Model, Batch 16, CPU):**
- Inference: 6.25 MB
- Training: 6.25 MB
- **Training/Inference Ratio:** 1.00x

### Key Observations
1. PyTorch has a lower overall memory footprint (60% less than TensorFlow on average)
2. TensorFlow shows consistent memory usage regardless of batch size
3. GPU execution provides ~4.3x memory efficiency compared to CPU for both frameworks
4. PyTorch shows linear scaling with batch size, while TensorFlow pre-allocates memory

## 💡 Practical Recommendations

Based on our analysis, we recommend:

1. **Framework Selection:**
   - For memory-constrained environments, PyTorch may be preferable
   - For applications with variable batch sizes, TensorFlow offers more predictable memory usage

2. **Hardware Considerations:**
   - GPU deployment offers significant memory benefits (~4.3x) for both frameworks
   - Consider the memory-computation tradeoffs when selecting hardware

3. **Batch Size Optimization:**
   - When scaling to larger batch sizes, consider TensorFlow's fixed memory allocation advantage
   - For PyTorch, be aware of the linear memory scaling with batch size

4. **Deployment Strategy:**
   - For edge devices, consider model size and framework memory footprint carefully
   - For cloud deployments, optimize for throughput while monitoring memory usage

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Yaxita Amin - [GitHub Profile](https://github.com/YaxitaAmin)

Project Link: [https://github.com/YaxitaAmin/Virtual_memory_profiling_MLmodels](https://github.com/YaxitaAmin/Virtual_memory_profiling_MLmodels)
