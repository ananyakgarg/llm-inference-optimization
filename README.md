
# LLM Inference Optimization: Quantization for Efficient Large Language Models

## Overview

This project explores state-of-the-art optimization techniques—specifically **dynamic quantization** to accelerate inference for large language models (LLMs). By comparing baseline full‑precision inference against optimized models on an M1 Mac, we aim to reduce latency and memory consumption while preserving output quality. The work demonstrates practical applications of model compression strategies for real-world deployment in resource-constrained environments.

## Motivation

Large language models have achieved remarkable results in NLP but are notorious for high inference costs. This project:
- **Reduces latency:** By compressing model weights and streamlining computations.
- **Decreases memory footprint:** Through lower-precision operations and sparse weight representations.
- **Maintains quality:** Balancing performance with minimal impact on output fidelity.

The work is particularly relevant for deploying AI services on edge devices or platforms with limited resources.

## Key Features

- **Dynamic Quantization Pipeline**: Implements INT8 quantization using PyTorch's `qnnpack` engine, specifically optimized for transformer architectures
- **Automated Benchmarking Suite**: Comprehensive performance analysis across multiple metrics:
  - Inference latency (mean and standard deviation)
  - Memory utilization tracking
  - Token generation throughput
- **Multi-Modal Inference Support**: Optimized for various compute environments:
  - CPU optimization with `qnnpack`
  - Automatic device selection (CPU/MPS)
  - Memory-efficient inference patterns

## Performance Metrics

The framework provides extensive benchmarking capabilities:
- Warm-up run elimination to ensure accurate timing
- Statistical analysis over multiple inference runs (n=20 by default)
- Memory tracking at the process level
- Token generation speed analysis

### Raw Inference Time
Below shows the raw, averaged inference time for the three different models. As we can see, the quantized-aware training method was the fastest. 
```
==================================================
Benchmark Summary:
Baseline Model:       300.6741 sec (Std: 384.2307 sec)
Dynamic Quantization: 276.1366 sec (Std: 553.7379 sec)
QAT Quantized:        101.7277 sec (Std: 110.5416 sec)
==================================================

```
and below are inference times with perplexity. 20 runs for each model (base, quantized, QAT) are averaged. 
```
==================================================
Benchmark Summary:
Baseline Model:       294.2346 sec (Std: 437.3341 sec), Perplexity: 32.80
Dynamic Quantization: 35.8609 sec (Std: 22.4715 sec), Perplexity: 32.82
QAT Quantized:        56.3815 sec (Std: 26.6446 sec), Perplexity: 32.80
==================================================
```

## Technical Architecture

### Core Components

1. **Baseline Inference Module** (`inference.py`)
```python
def run_baseline_inference(
    model_name="gpt2-large",
    temperature=0.7,
    top_k=30,
    top_p=0.85,
    repetition_penalty=1.2
)
```
- Implements optimized inference parameters
- Automatic device selection (CPU/MPS)
- Memory-efficient execution patterns
- Process-level memory tracking

2. **Quantization Engine** (`quantization.py`)
```python
def apply_dynamic_quantization(model="gpt2-large"):
    return torch.ao.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
```
- Dynamic quantization of linear layers to INT8
- Selective layer quantization support
- Automated model export and persistence

3. **Benchmarking System** (`benchmark.py`)
```python
def benchmark_inference(
    model,
    tokenizer,
    prompt,
    n_runs=20,
    max_length=150
)
```
- Statistical performance analysis
- Warm-up run execution
- Standard deviation calculation
- Token generation tracking

## 🔧 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-inference-optimization.git

# Navigate to project directory
cd llm-inference-optimization

# Install dependencies
pip install torch transformers psutil
```

## Usage

### Running Baseline Inference

```bash
./scripts/run_baseline.sh
```

### Applying Quantization

```bash
./scripts/run_quantization.sh
```

### Running Benchmarks

```python
python src/benchmark.py
```

## Implementation Details

### Optimization Techniques

1. **Dynamic Quantization**
   - INT8 quantization for linear layers
   - Selective layer quantization
   - `qnnpack` engine optimization

2. **Memory Management**
   - Explicit cache clearing for MPS devices
   - Process-level memory tracking
   - Efficient tensor management

3. **Generation Parameters**
   - Optimized temperature (0.7)
   - Balanced top-k (30) and top-p (0.85)
   - Repetition penalty implementation (1.2)

## Project Structure

```
├── models/
│   └── quantized_gpt2/
│       └── state_dict.pth
├── scripts/
│   ├── run_baseline.sh
│   └── run_quantization.sh
└── src/
    ├── benchmark.py
    ├── inference.py
    ├── quantization.py
```

## Technical Dependencies

- PyTorch >= 1.13.0
- Transformers >= 4.25.1
- Python >= 3.8
- QNNPACK backend support

## Contributing

Contributions are welcome! Areas of particular interest:
- Additional quantization techniques
- Multi-GPU support
- Extended benchmarking metrics
- Alternative model architecture support
