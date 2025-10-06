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
- **Data**: Input data is **duplicated** across all GPUs
- **Weights**: Output features are **split** across GPUs
- Each GPU computes partial output (50 features)
- Uses `all_gather` to concatenate results
- Communication: Only gradient synchronization

### Column-wise (Vertical) Splitting  
- **Data**: Input data is **split** across GPUs
- **Weights**: Input features are **split** across GPUs
- Each GPU processes different input slice
- Uses `all_reduce` to sum partial results
- Communication: Forward pass + gradient synchronization

## Requirements

- PyTorch with CUDA support
- 2+ GPUs
- NCCL backend (automatically configured)

## Observations

### Performance Characteristics
- **Row-wise splitting** typically shows 2-3x better performance than column-wise
- **Communication overhead** is the primary bottleneck in distributed training
- **All-gather operations** generally have lower latency than all-reduce operations
- **Throughput** can vary significantly based on tensor sizes and communication patterns

### Communication Patterns
- **NCCL Ring Algorithm**: Uses asymmetric ring topology where different GPUs have different workloads
- **All-gather timing**: Ring leader (typically rank 0) completes faster than ring followers
- **All-reduce timing**: More symmetric but still shows slight variations between ranks
- **Memory access patterns**: Different GPUs may show varying memory bandwidth utilization

### Profiling Insights
- **Forward pass**: Usually the most time-consuming operation
- **Communication kernels**: NCCL operations show varying durations across ranks
- **Gradient synchronization**: Adds significant overhead to training time
- **CUDA context switching**: Can cause timing variations between GPUs

*Note: Specific performance numbers vary based on hardware, tensor sizes, and system configuration.*