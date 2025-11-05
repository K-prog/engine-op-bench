#!/usr/bin/env python3
"""
AI Engine Profiling Script
Profiles CoreML (CPU, ANE, GPU), ONNX (CPU, DML), and OpenVINO (GPU, NPU)
across various ML operators with increasing computational complexity.
"""

import numpy as np
import time
import csv
import platform
import sys
from datetime import datetime
from pathlib import Path
import argparse

# Engine-specific imports with error handling
ENGINES_AVAILABLE = {
    'coreml': False,
    'onnx': False,
    'openvino': False
}

try:
    import coremltools as ct
    ENGINES_AVAILABLE['coreml'] = True
except ImportError:
    print("Warning: CoreML not available")

try:
    import onnxruntime as ort
    ENGINES_AVAILABLE['onnx'] = True
except ImportError:
    print("Warning: ONNX Runtime not available")

try:
    from openvino.runtime import Core
    ENGINES_AVAILABLE['openvino'] = True
except ImportError:
    print("Warning: OpenVINO not available")


class OperatorProfiler:
    """Base class for profiling ML operators"""
    
    def __init__(self, warmup_runs=5, benchmark_runs=50):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
        
    # def generate_conv_configs(self):
    #     """Generate convolution configurations with increasing complexity"""
    #     configs = []
    #     # Small to large models
    #     for batch in [1, 4]:
    #         for channels_in, channels_out in [(3, 32), (64, 128), (256, 512)]:
    #             for kernel_size in [3]:
    #                 for image_size in [32, 128, 512]:
    #                     configs.append({
    #                         'name': 'conv2d',
    #                         'batch': batch,
    #                         'channels_in': channels_in,
    #                         'channels_out': channels_out,
    #                         'kernel_size': kernel_size,
    #                         'height': image_size,
    #                         'width': image_size,
    #                         'stride': 1,
    #                         'padding': kernel_size // 2
    #                     })
    #     return configs
    
    def generate_conv_configs(self):
        """Generate convolution configurations with increasing complexity"""
        configs = []
        # Small to large models
        for batch in [1, 4]:
            for channels_in, channels_out in [(3, 32), (64, 128), (256, 512)]:
                for kernel_size in [3]:
                    for image_size in [32, 128, 512]:
                        configs.append({
                            'name': 'conv2d',
                            'batch': batch,
                            'channels_in': channels_in,
                            'channels_out': channels_out,
                            'kernel_size': kernel_size,
                            'height': image_size,
                            'width': image_size,
                            'stride': 1,
                            'padding': kernel_size // 2
                        })
        return configs
    
    def generate_matmul_configs(self):
        """Generate matrix multiplication configurations"""
        configs = []
        for batch in [1, 4, 8]:
            for m, n, k in [(64, 64, 64), (256, 256, 256), (1024, 1024, 1024)]:
                configs.append({
                    'name': 'matmul',
                    'batch': batch,
                    'm': m,
                    'n': n,
                    'k': k
                })
        return configs
    
    def generate_pooling_configs(self):
        """Generate pooling configurations"""
        configs = []
        for pool_type in ['max', 'avg']:
            for batch in [1, 4]:
                for channels in [64, 256, 512]:
                    for image_size in [32, 128, 224]:
                        for kernel_size in [2, 3]:
                            configs.append({
                                'name': f'{pool_type}_pool',
                                'batch': batch,
                                'channels': channels,
                                'height': image_size,
                                'width': image_size,
                                'kernel_size': kernel_size,
                                'stride': 2
                            })
        return configs
    
    def generate_activation_configs(self):
        """Generate activation function configurations"""
        configs = []
        for activation in ['relu', 'gelu', 'sigmoid', 'tanh']:
            for batch in [1, 4, 16]:
                for size in [1024, 4096, 65536]:
                    configs.append({
                        'name': activation,
                        'batch': batch,
                        'size': size
                    })
        return configs
    
    def generate_batchnorm_configs(self):
        """Generate batch normalization configurations"""
        configs = []
        for batch in [1, 4, 16]:
            for channels in [32, 128, 512]:
                for image_size in [32, 64, 224]:
                    configs.append({
                        'name': 'batchnorm',
                        'batch': batch,
                        'channels': channels,
                        'height': image_size,
                        'width': image_size
                    })
        return configs
    
    def calculate_flops(self, config):
        """Calculate approximate FLOPs for an operation"""
        name = config['name']
        
        if name == 'conv2d':
            # FLOPs = 2 * batch * out_channels * out_height * out_width * in_channels * kernel_h * kernel_w
            out_h = config['height']
            out_w = config['width']
            flops = (2 * config['batch'] * config['channels_out'] * out_h * out_w * 
                    config['channels_in'] * config['kernel_size'] * config['kernel_size'])
        elif name == 'matmul':
            # FLOPs = 2 * batch * m * n * k
            flops = 2 * config['batch'] * config['m'] * config['n'] * config['k']
        elif 'pool' in name:
            # Pooling: batch * channels * out_h * out_w * kernel^2
            out_h = config['height'] // config['stride']
            out_w = config['width'] // config['stride']
            flops = config['batch'] * config['channels'] * out_h * out_w * config['kernel_size']**2
        elif name in ['relu', 'gelu', 'sigmoid', 'tanh']:
            # Element-wise ops
            flops = config['batch'] * config['size']
        elif name == 'batchnorm':
            # BatchNorm: ~2 ops per element (subtract mean, divide by std)
            flops = 2 * config['batch'] * config['channels'] * config['height'] * config['width']
        else:
            flops = 0
        
        return flops


class CoreMLProfiler(OperatorProfiler):
    """Profiler for CoreML engine"""
    
    def __init__(self, compute_unit='ALL', **kwargs):
        super().__init__(**kwargs)
        self.compute_unit = compute_unit
        
    def profile_operator(self, config):
        """Profile a single operator configuration"""
        try:
            import torch
            import torch.nn as nn
            
            name = config['name']
            
            # Create PyTorch model based on config
            if name == 'conv2d':
                model = nn.Conv2d(config['channels_in'], config['channels_out'], 
                                 config['kernel_size'], config['stride'], config['padding'])
                input_shape = (config['batch'], config['channels_in'], config['height'], config['width'])
            elif name == 'matmul':
                model = nn.Linear(config['k'], config['n'], bias=False)
                input_shape = (config['batch'], config['m'], config['k'])
            elif 'pool' in name:
                if 'max' in name:
                    model = nn.MaxPool2d(config['kernel_size'], config['stride'])
                else:
                    model = nn.AvgPool2d(config['kernel_size'], config['stride'])
                input_shape = (config['batch'], config['channels'], config['height'], config['width'])
            elif name == 'relu':
                model = nn.ReLU()
                input_shape = (config['batch'], config['size'])
            elif name == 'gelu':
                model = nn.GELU()
                input_shape = (config['batch'], config['size'])
            elif name == 'sigmoid':
                model = nn.Sigmoid()
                input_shape = (config['batch'], config['size'])
            elif name == 'tanh':
                model = nn.Tanh()
                input_shape = (config['batch'], config['size'])
            elif name == 'batchnorm':
                model = nn.BatchNorm2d(config['channels'])
                input_shape = (config['batch'], config['channels'], config['height'], config['width'])
            else:
                return None
            
            model.eval()
            example_input = torch.randn(*input_shape)
            
            # Convert to CoreML
            traced_model = torch.jit.trace(model, example_input)
            
            if self.compute_unit == 'CPU':
                compute_units = ct.ComputeUnit.CPU_ONLY
            elif self.compute_unit == 'ANE':
                compute_units = ct.ComputeUnit.ALL  # ANE included in ALL
            elif self.compute_unit == 'GPU':
                compute_units = ct.ComputeUnit.CPU_AND_GPU
            else:
                compute_units = ct.ComputeUnit.ALL
            
            mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(shape=input_shape, name = "input")],
                                compute_units=compute_units,
                                convert_to = "mlprogram",
                                minimum_deployment_target = ct.target.macOS13)
            
            # Prepare input
            input_dict = {"input": example_input.numpy()}
            
            # Warmup
            for _ in range(self.warmup_runs):
                mlmodel.predict(input_dict)
            
            # Benchmark
            times = []
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                mlmodel.predict(input_dict)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            
            return {
                'mean_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times)
            }
            
        except Exception as e:
            print(f"Error profiling {config['name']} on CoreML {self.compute_unit}: {e}")
            return None


class ONNXProfiler(OperatorProfiler):
    """Profiler for ONNX Runtime"""
    
    def __init__(self, provider='CPUExecutionProvider', **kwargs):
        super().__init__(**kwargs)
        self.provider = provider
        
    def profile_operator(self, config):
        """Profile a single operator configuration"""
        try:
            import torch
            import torch.nn as nn
            
            name = config['name']
            
            # Create PyTorch model
            if name == 'conv2d':
                model = nn.Conv2d(config['channels_in'], config['channels_out'], 
                                 config['kernel_size'], config['stride'], config['padding'])
                input_shape = (config['batch'], config['channels_in'], config['height'], config['width'])
            elif name == 'matmul':
                model = nn.Linear(config['k'], config['n'], bias=False)
                input_shape = (config['batch'], config['m'], config['k'])
            elif 'pool' in name:
                if 'max' in name:
                    model = nn.MaxPool2d(config['kernel_size'], config['stride'])
                else:
                    model = nn.AvgPool2d(config['kernel_size'], config['stride'])
                input_shape = (config['batch'], config['channels'], config['height'], config['width'])
            elif name == 'relu':
                model = nn.ReLU()
                input_shape = (config['batch'], config['size'])
            elif name == 'gelu':
                model = nn.GELU()
                input_shape = (config['batch'], config['size'])
            elif name == 'sigmoid':
                model = nn.Sigmoid()
                input_shape = (config['batch'], config['size'])
            elif name == 'tanh':
                model = nn.Tanh()
                input_shape = (config['batch'], config['size'])
            elif name == 'batchnorm':
                model = nn.BatchNorm2d(config['channels'])
                input_shape = (config['batch'], config['channels'], config['height'], config['width'])
            else:
                return None
            
            model.eval()
            example_input = torch.randn(*input_shape)
            
            # Export to ONNX
            onnx_path = "/tmp/temp_model.onnx"
            torch.onnx.export(model, example_input, onnx_path, 
                            input_names=['input'], output_names=['output'],
                            opset_version=13)
            
            # Create ONNX Runtime session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(onnx_path, sess_options, 
                                          providers=[self.provider])
            
            input_data = example_input.numpy()
            
            # Warmup
            for _ in range(self.warmup_runs):
                session.run(None, {'input': input_data})
            
            # Benchmark
            times = []
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                session.run(None, {'input': input_data})
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            return {
                'mean_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times)
            }
            
        except Exception as e:
            print(f"Error profiling {config['name']} on ONNX {self.provider}: {e}")
            return None


class OpenVINOProfiler(OperatorProfiler):
    """Profiler for OpenVINO"""
    
    def __init__(self, device='CPU', **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.core = Core()
        
    def profile_operator(self, config):
        """Profile a single operator configuration"""
        try:
            import torch
            import torch.nn as nn
            
            name = config['name']
            
            # Create PyTorch model
            if name == 'conv2d':
                model = nn.Conv2d(config['channels_in'], config['channels_out'], 
                                 config['kernel_size'], config['stride'], config['padding'])
                input_shape = (config['batch'], config['channels_in'], config['height'], config['width'])
            elif name == 'matmul':
                model = nn.Linear(config['k'], config['n'], bias=False)
                input_shape = (config['batch'], config['m'], config['k'])
            elif 'pool' in name:
                if 'max' in name:
                    model = nn.MaxPool2d(config['kernel_size'], config['stride'])
                else:
                    model = nn.AvgPool2d(config['kernel_size'], config['stride'])
                input_shape = (config['batch'], config['channels'], config['height'], config['width'])
            elif name == 'relu':
                model = nn.ReLU()
                input_shape = (config['batch'], config['size'])
            elif name == 'gelu':
                model = nn.GELU()
                input_shape = (config['batch'], config['size'])
            elif name == 'sigmoid':
                model = nn.Sigmoid()
                input_shape = (config['batch'], config['size'])
            elif name == 'tanh':
                model = nn.Tanh()
                input_shape = (config['batch'], config['size'])
            elif name == 'batchnorm':
                model = nn.BatchNorm2d(config['channels'])
                input_shape = (config['batch'], config['channels'], config['height'], config['width'])
            else:
                return None
            
            model.eval()
            example_input = torch.randn(*input_shape)
            
            # Export to ONNX first
            onnx_path = "/tmp/temp_model_ov.onnx"
            torch.onnx.export(model, example_input, onnx_path,
                            input_names=['input'], output_names=['output'],
                            opset_version=13)
            
            # Convert to OpenVINO IR
            ov_model = self.core.read_model(onnx_path)
            compiled_model = self.core.compile_model(ov_model, self.device)
            
            input_data = example_input.numpy()
            
            # Warmup
            for _ in range(self.warmup_runs):
                compiled_model([input_data])
            
            # Benchmark
            times = []
            for _ in range(self.benchmark_runs):
                start = time.perf_counter()
                compiled_model([input_data])
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            return {
                'mean_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times)
            }
            
        except Exception as e:
            print(f"Error profiling {config['name']} on OpenVINO {self.device}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Profile AI engines across ML operators')
    parser.add_argument('--output', '-o', default='profiling_results.csv', 
                       help='Output CSV file name')
    parser.add_argument('--operators', '-op', nargs='+', 
                       choices=['conv', 'matmul', 'pool', 'activation', 'batchnorm', 'all'],
                       default=['all'], help='Operators to profile')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup runs')
    parser.add_argument('--benchmark', type=int, default=50, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # System information
    system_info = {
        'hostname': platform.node(),
        'system': platform.system(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'timestamp': datetime.now().isoformat()
    }
    
    print("="*80)
    print("AI Engine Profiling Tool")
    print("="*80)
    print(f"System: {system_info['system']}")
    print(f"Processor: {system_info['processor']}")
    print(f"Hostname: {system_info['hostname']}")
    print(f"Timestamp: {system_info['timestamp']}")
    print("="*80)
    print(f"Available engines: {[k for k, v in ENGINES_AVAILABLE.items() if v]}")
    print("="*80)
    
    # Generate configurations
    profiler = OperatorProfiler(warmup_runs=args.warmup, benchmark_runs=args.benchmark)
    
    all_configs = []
    if 'all' in args.operators or 'conv' in args.operators:
        all_configs.extend(profiler.generate_conv_configs())
    if 'all' in args.operators or 'matmul' in args.operators:
        all_configs.extend(profiler.generate_matmul_configs())
    if 'all' in args.operators or 'pool' in args.operators:
        all_configs.extend(profiler.generate_pooling_configs())
    if 'all' in args.operators or 'activation' in args.operators:
        all_configs.extend(profiler.generate_activation_configs())
    if 'all' in args.operators or 'batchnorm' in args.operators:
        all_configs.extend(profiler.generate_batchnorm_configs())
    
    print(f"Total configurations to test: {len(all_configs)}")
    print("="*80)
    
    # Prepare results file
    results = []
    
    # Define engine configurations
    engine_configs = []
    
    if ENGINES_AVAILABLE['openvino']:
        ov_core = Core()
        available_devices = ov_core.available_devices
        if 'GPU' in available_devices:
            engine_configs.append(
                ('OpenVINO', 'GPU', OpenVINOProfiler(device='GPU', warmup_runs=args.warmup, benchmark_runs=args.benchmark))
            )
        if 'NPU' in available_devices:
            engine_configs.append(
                ('OpenVINO', 'NPU', OpenVINOProfiler(device='NPU', warmup_runs=args.warmup, benchmark_runs=args.benchmark))
            )
    
    # Profile each configuration
    total_tests = len(all_configs) * len(engine_configs)
    current_test = 0
    
    for config in all_configs:
        flops = profiler.calculate_flops(config)
        
        for engine_name, device, engine_profiler in engine_configs:
            current_test += 1
            print(f"[{current_test}/{total_tests}] Testing {engine_name}-{device}: {config['name']} "
                  f"(FLOPs: {flops/1e6:.2f}M)", end='... ')
            
            result = engine_profiler.profile_operator(config)
            
            if result:
                result_entry = {
                    'system': system_info['hostname'],
                    'timestamp': system_info['timestamp'],
                    'engine': engine_name,
                    'device': device,
                    'operator': config['name'],
                    'flops': flops,
                    'config': str(config),
                    **result
                }
                results.append(result_entry)
                print(f"✓ {result['mean_time_ms']:.3f}ms")
            else:
                print("✗ Failed")
    
    # Write results to CSV
    if results:
        fieldnames = ['system', 'timestamp', 'engine', 'device', 'operator', 'flops', 
                     'mean_time_ms', 'std_time_ms', 'min_time_ms', 'max_time_ms', 'config']
        
        with open(args.output, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print("="*80)
        print(f"Results saved to: {args.output}")
        print(f"Total successful tests: {len(results)}")
        print("="*80)
    else:
        print("No results collected!")


if __name__ == '__main__':
    main()