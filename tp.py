import torch
import torch.distributed as dist
import torch.nn.functional as F
import os
import time
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

# Global variables for timing comparison
row_wise_time = 0
column_wise_time = 0

class TensorParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, rank, world_size, device, mode="row"):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.mode = mode
        
        if mode == "row":
            # Row-wise splitting (horizontal)
            features_per_rank = out_features // world_size
            start = rank * features_per_rank
            end = (rank + 1) * features_per_rank
            
            self.W = torch.nn.Parameter(torch.randn(features_per_rank, in_features, device=device) * 0.02)
            self.b = torch.nn.Parameter(torch.zeros(features_per_rank, device=device))
            
        elif mode == "column":
            # Column-wise splitting (vertical)
            features_per_rank = in_features // world_size
            start = rank * features_per_rank
            end = (rank + 1) * features_per_rank
            
            self.W = torch.nn.Parameter(torch.randn(out_features, features_per_rank, device=device) * 0.02)
            self.b = torch.nn.Parameter(torch.zeros(out_features, device=device))
            
            self.start = start
            self.end = end

    def forward(self, x):
        if self.mode == "row":
            # Row-wise: each rank computes partial output, then all_gather
            y_partial = x @ self.W.T + self.b
            
            full_output = torch.zeros(x.shape[0], self.world_size * y_partial.shape[1], 
                                    device=y_partial.device, dtype=y_partial.dtype, requires_grad=True)
            dist.all_gather_into_tensor(full_output, y_partial)
            return full_output
            
        elif self.mode == "column":
            # Column-wise: split input, compute partial, then all_reduce
            x_local = x[:, self.start:self.end]
            
            y_partial = x_local @ self.W.T
            dist.all_reduce(y_partial, op=dist.ReduceOp.SUM)
            return y_partial + self.b

def train_loop(rank, world_size, mode="row", init_distributed=True, enable_profiler=False):
    global row_wise_time, column_wise_time
    
    if init_distributed:
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    model = TensorParallelLinear(1000 , 1000, rank, world_size, device, mode).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    
    data = torch.randn(20, 1000, device=device, requires_grad=True)
    targets = torch.randn(20, 1000, device=device)
    
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        optimizer.step()
    
    # Setup profiler
    if enable_profiler:
        profiler_dir = f"profiler_logs_rank_{rank}_{mode}"
        os.makedirs(profiler_dir, exist_ok=True)
        
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            record_shapes=True,
            with_stack=True
        )
        profiler.start()
    
    # Timing measurement
    torch.cuda.synchronize()
    start_time = time.time()
    
    for epoch in range(10):
        with record_function(f"epoch_{epoch}"):
            optimizer.zero_grad()
            
            with record_function("forward_pass"):
                outputs = model(data)
                loss = F.mse_loss(outputs, targets)
            
            with record_function("backward_pass"):
                loss.backward()
            
            with record_function("gradient_sync"):
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad /= world_size
            
            with record_function("optimizer_step"):
                optimizer.step()
        
        if enable_profiler:
            profiler.step()
        
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    if enable_profiler:
        profiler.stop()
        profiler_dir = f"profiler_logs_rank_{rank}_{mode}"
        if rank == 0:
            print(f"\nProfiler data saved to {profiler_dir}/")
            print(f"View with: tensorboard --logdir={profiler_dir}")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    if rank == 0:
        total_time = end_time - start_time
        print(f"\n{mode.upper()}-WISE Results:")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Time per epoch: {total_time/10:.4f} seconds")
        print(f"Throughput: {10/total_time:.2f} epochs/second")
        
        if mode == "row":
            global row_wise_time
            row_wise_time = total_time
        elif mode == "column":
            global column_wise_time
            column_wise_time = total_time
    
    if init_distributed:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["row", "column", "both"], 
                       help="Tensor parallel mode: row-wise, column-wise, or both")
    parser.add_argument("--profile", action="store_true", 
                       help="Enable PyTorch profiler")
    args = parser.parse_args()
    
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if args.profile and rank == 0:
        print("Profiling enabled - data will be saved to current directory")
    
    if args.mode == "both":
        if rank == 0:
            print("Running both tensor parallel modes")
            print("=" * 50)
        
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        if rank == 0:
            print("\nRow-wise (all_gather) Tensor Parallelism:")
            print("-" * 40)
        train_loop(rank, world_size, "row", init_distributed=False, enable_profiler=args.profile)
        
        time.sleep(1)
        
        if rank == 0:
            print("\nColumn-wise (all_reduce) Tensor Parallelism:")
            print("-" * 40)
        train_loop(rank, world_size, "column", init_distributed=False, enable_profiler=args.profile)
        
        dist.destroy_process_group()
        
        if rank == 0:
            print("\nPerformance Comparison:")
            print("=" * 60)
            print(f"{'Metric':<20} {'Row-wise':<15} {'Column-wise':<15}")
            print("-" * 50)
            print(f"{'Total Time (s)':<20} {f'{row_wise_time:.4f}':<15} {f'{column_wise_time:.4f}':<15}")
            print(f"{'Throughput (eps/s)':<20} {f'{10/row_wise_time:.2f}':<15} {f'{10/column_wise_time:.2f}':<15}")
            print(f"{'Speedup':<20} {'1.00x':<15} {f'{row_wise_time/column_wise_time:.2f}x':<15}")
            improvement = ((10/row_wise_time) / (10/column_wise_time) - 1) * 100
            print(f"\nRow-wise is {improvement:.0f}% faster than column-wise")
            print("=" * 60)
            
            if args.profile:
                print(f"\nProfiler data saved to current directory")
                print("View with: tensorboard --logdir=profiler_logs_rank_0_row")
    else:
        train_loop(rank, world_size, args.mode, enable_profiler=args.profile)

if __name__ == "__main__":
    main()