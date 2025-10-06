# Tensor Parallel Training Benchmark

A clean benchmarking implementation of tensor parallel training using PyTorch distributed with both row-wise and column-wise splitting strategies.

## Features

- **Row-wise Tensor Parallelism**: Horizontal splitting using `all_gather` communication
- **Column-wise Tensor Parallelism**: Vertical splitting using `all_reduce` communication  
- **Performance Comparison**: Side-by-side timing and throughput analysis
- **PyTorch Profiler Integration**: Detailed performance analysis with TensorBoard visualization
- **NCCL Backend**: Optimized for GPU communication with container compatibility

## Usage

```bash
# Run both modes for comparison
torchrun --nproc_per_node=2 tp.py --mode both

# Run specific mode
torchrun --nproc_per_node=2 tp.py --mode row
torchrun --nproc_per_node=2 tp.py --mode column

# Enable PyTorch Profiler (saves data to current directory)
torchrun --nproc_per_node=2 tp.py --mode both --profile

# View profiler data with TensorBoard
tensorboard --logdir=profiler_logs_rank_0_row
```

## Implementation Details

### Row-wise (Horizontal) Splitting
- Splits output features across GPUs
- Each GPU computes partial output (50 features)
- Uses `all_gather` to concatenate results
- Communication: Only gradient synchronization

### Column-wise (Vertical) Splitting  
- Splits input features across GPUs
- Each GPU processes different input slice
- Uses `all_reduce` to sum partial results
- Communication: Forward pass + gradient synchronization

## Requirements

- PyTorch with CUDA support
- 2+ GPUs
- NCCL backend (automatically configured)