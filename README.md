# AI Engine Profiling System

A comprehensive benchmarking suite for profiling different AI inference engines (CoreML, ONNX Runtime, OpenVINO) across various hardware accelerators and ML operators.

## Overview

This system helps you:
- Profile ML operators (Conv2D, MatMul, Pooling, Activations, BatchNorm) with increasing computational complexity
- Compare performance across different AI engines and hardware accelerators
- Visualize how engines scale from small to large models
- Aggregate results from multiple systems for comparison

## Features

### Profiling Script (`profile_ai_engines.py`)
- **Engines Supported:**
  - CoreML: CPU, ANE (Apple Neural Engine), GPU
  - ONNX Runtime: CPU, DML (DirectML for Windows)
  - OpenVINO: CPU, GPU, NPU

- **Operators Tested:**
  - Convolution (Conv2D) - various kernel sizes, channels, image sizes
  - Matrix Multiplication (MatMul/Linear) - different dimensions
  - Pooling (MaxPool, AvgPool)
  - Activations (ReLU, GELU, Sigmoid, Tanh)
  - Batch Normalization

- **Complexity Scaling:**
  - Tests range from small operations (few MFLOPs) to large operations (several GFLOPs)
  - Mimics scaling from mobile models to server-side models

### Visualization Script (`visualize_results.py`)
- **Line graphs** showing execution time vs computational complexity
- **Throughput plots** (GFLOPS) showing efficiency at different scales
- **Heatmaps** comparing operators across engines
- **System comparisons** when multiple CSVs are provided
- **Summary statistics** in JSON and CSV formats

## Installation

### Common Requirements
```bash
pip install numpy pandas matplotlib seaborn torch
```

### Engine-Specific Requirements

#### CoreML (macOS only)
```bash
pip install coremltools
```

#### ONNX Runtime
```bash
# CPU version
pip install onnxruntime

# Or GPU version (if you have CUDA)
pip install onnxruntime-gpu

# For Windows with DirectML
pip install onnxruntime-directml
```

#### OpenVINO
```bash
pip install openvino
```

## Usage

### 1. Run Profiling on Each System

**Basic usage (all operators):**
```bash
python profile_ai_engines.py -o results_system1.csv
```

**Profile specific operators:**
```bash
python profile_ai_engines.py -o results_laptop.csv --operators conv matmul
```

**Adjust benchmark parameters:**
```bash
python profile_ai_engines.py -o results.csv --warmup 10 --benchmark 100
```

**Available options:**
- `--output, -o`: Output CSV filename (default: profiling_results.csv)
- `--operators, -op`: Choose operators to test: conv, matmul, pool, activation, batchnorm, all
- `--warmup`: Number of warmup runs (default: 5)
- `--benchmark`: Number of benchmark runs (default: 50)

### 2. Collect CSVs from Multiple Systems

Run the profiling script on each system you want to compare:
- Your development laptop
- A workstation with GPU
- A server with NPU
- Different OS (Windows, macOS, Linux)

Name each CSV file distinctively:
```
results_macbook_m1.csv
results_windows_rtx4090.csv
results_linux_npu.csv
```

### 3. Visualize Results

**Basic usage (all CSVs in current directory):**
```bash
python visualize_results.py
```

**Specify input pattern:**
```bash
python visualize_results.py --input "results_*.csv"
```

**Specify output directory:**
```bash
python visualize_results.py --output-dir my_visualizations
```

**Change output format:**
```bash
python visualize_results.py --format pdf
```

### Output Files

The visualization script generates:

1. **Per-Operator Scaling Plots** (`scaling_<operator>.png`)
   - Shows how each engine scales with increasing FLOPs
   - Separate plots for execution time and throughput

2. **Overall Engine Comparison** (`engine_comparison_overall.png`)
   - Combined view of all engines across all operators
   - Includes throughput distribution and efficiency metrics

3. **Operator Heatmap** (`operator_heatmap.png`)
   - Color-coded performance matrix
   - Easy to spot strengths and weaknesses

4. **System Comparison** (`system_comparison.png`)
   - Only generated when multiple systems are present
   - Side-by-side comparison of different hardware

5. **Summary Statistics**
   - `summary_statistics.json`: Overall stats and top performers
   - `engine_statistics.csv`: Detailed per-engine metrics
   - `operator_statistics.csv`: Detailed per-operator metrics

## Understanding the Results

### Metrics Explained

- **FLOPs**: Floating-point operations (theoretical work)
- **Execution Time (ms)**: Wall-clock time to execute the operation
- **Throughput (GFLOPS)**: Billion FLOPs per second - higher is better
- **Efficiency**: GFLOPS per millisecond - computational efficiency

### Interpreting the Graphs

1. **Execution Time vs FLOPs (Log-Log Scale)**
   - Steeper slope = worse scaling
   - Flat line = excellent scaling
   - Lines that diverge at high FLOPs indicate poor large-model performance

2. **Throughput vs FLOPs**
   - Higher lines = better performance
   - Increasing throughput = good scaling (engine getting more efficient)
   - Decreasing throughput = poor scaling (hitting bottlenecks)

3. **Heatmap**
   - Red = high throughput (good)
   - Yellow = medium throughput
   - Light colors = low throughput (poor)

## Example Workflow

```bash
# On MacBook M1
python profile_ai_engines.py -o results_m1.csv

# On Windows PC with RTX GPU
python profile_ai_engines.py -o results_windows_rtx.csv

# On Linux server with Intel NPU
python profile_ai_engines.py -o results_linux_npu.csv

# Aggregate and visualize all results
python visualize_results.py --input "results_*.csv" --output-dir comparison_charts
```

## Tips for Best Results

1. **Close other applications** during profiling for consistent results
2. **Ensure adequate cooling** - thermal throttling affects results
3. **Use AC power** on laptops (not battery)
4. **Run multiple times** and average if you see high variance
5. **Document your system specs** in the CSV filename for easy identification

## Troubleshooting

### "ImportError: No module named X"
Install the missing engine: `pip install <engine-package>`

### "No engines available"
At least one engine must be installed. Install CoreML, ONNX Runtime, or OpenVINO.

### "CUDA/GPU not found" 
Make sure you have the GPU version of the package and proper drivers installed.

### High variance in results
Increase `--benchmark` runs: `--benchmark 100`

### Memory errors with large operations
The script automatically handles this by skipping failed operations and continuing.

## CSV Format

Each CSV contains:
- `system`: Hostname of the system
- `timestamp`: When the test was run
- `engine`: AI engine name (CoreML, ONNX, OpenVINO)
- `device`: Hardware accelerator (CPU, GPU, ANE, NPU, DML)
- `operator`: Operation type
- `flops`: Theoretical floating-point operations
- `mean_time_ms`: Average execution time
- `std_time_ms`: Standard deviation
- `min_time_ms`: Minimum time
- `max_time_ms`: Maximum time
- `config`: Full configuration dictionary

## License

MIT License - feel free to use and modify as needed.

## Contributing

Suggestions and improvements welcome! Areas for enhancement:
- Additional operators (LayerNorm, Attention, etc.)
- More engines (TensorRT, TFLite, etc.)
- Power consumption measurements
- Batch size effects
- Precision comparisons (FP32 vs FP16 vs INT8)