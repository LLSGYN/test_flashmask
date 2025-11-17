import os
import numpy as np
from functools import lru_cache
from typing import Optional, List
import random

# from block_sparse_attn import (
#     block_sparse_attn_func,
#     flash_attn_varlen_func,
# )

import paddle
import paddle.nn.functional as F
from paddle.nn.functional.flash_attention import flashmask_attention

from tabulate import tabulate



# paddle.manual_seed(0)

np.random.seed(0)
random.seed(0)

# paddle._dynamo.config.cache_size_limit = 1000

def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = paddle.quantile(times, paddle.to_tensor(quantiles, dtype=paddle.float32)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times.tolist()
    return getattr(paddle, return_mode)(times).item()

@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=True, BLOCK_SIZE=[128, 128])
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12

def cal_flops(B, H, Sq, Sk, D, mode='fwd'):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def cal_tflops(flops, time_ms):
    return  flops * (1e3 / time_ms) / 1e12

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")

def generate_ones_mask(B, S, H, D):
    startend_row_indices = paddle.zeros(
        shape=(B, H, S, 2), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = False
    return startend_row_indices, causal


def generate_causal_mask(B,S,H,D):
    startend_row_indices = paddle.zeros(
        shape=(B, H, S, 1), dtype="int32"
    )
    startend_row_indices[:,:,:,0]=S
    causal = True
    return startend_row_indices, causal

def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", "median", or "all" Default is "mean".    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    #print("pt1")    
    fn()
    #print("pt2")

    paddle.device.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = paddle.empty([int(cache_size // 4)], dtype=paddle.int32)
    else:
        cache = paddle.empty([int(cache_size)], dtype=paddle.int8)

    # Estimate the runtime of the function
    #print("pt1")
    start_event = paddle.device.Event(enable_timing=True)
    end_event = paddle.device.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    #print("pt2")
    end_event.record()
    paddle.device.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    n_warmup = 10
    n_repeat = 50
    start_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [paddle.device.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        #cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    paddle.device.synchronize()
    times = paddle.to_tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=paddle.float32)
    return _summarize_statistics(times, quantiles, return_mode)

def test_block_mask(
    B: int = 16,
    H: int = 16,
    S: int = 8192,
    D: int = 64,
    dtype = 'bf16',
    skip_correctness: bool = False,
    print_mask: bool = True,
    device: str = "cuda",
    causal: bool = False,
    sparsity: float = 0.0,
):
    if dtype == 'bfloat16':
        data_type = paddle.bfloat16
    else:
        data_type = paddle.float16
        
    seqlen = S
    batch_size = B
    nheads = H
    headdim = D
    dropout_p = 0.0
    causal = causal
    shape = (batch_size , seqlen, nheads, headdim)

    q = paddle.randn(shape, device=device, dtype=data_type, requires_grad=True)
    k = paddle.randn(shape, device=device, dtype=data_type, requires_grad=True)
    v = paddle.randn(shape, device=device, dtype=data_type, requires_grad=True)
    gradOut = paddle.randn(shape, device=device, dtype=data_type)

    cu_seqlens = paddle.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=paddle.int32, device=device)
    head_mask_type = paddle.tensor([1] * nheads, device=device, dtype=paddle.int32)
    base_blockmask, real_sparsity = generate_base_sparsity_mask(seqlen, seqlen, block_size, block_size, block_size, sparsity, causal = causal, device=device)
    
    # print(base_blockmask)
    
    if causal:
        start_row_indices, causal = generate_causal_mask(B, S, H, D)
    else:
        start_row_indices, causal = generate_ones_mask(B, S, H, D)
    
    base_blockmask = base_blockmask.unsqueeze(0).repeat(start_row_indices.shape[0], start_row_indices.shape[1], 1, 1).astype(paddle.int32)
    flash_block_attention_call = lambda: flashmask_attention(q, k, v, startend_row_indices=start_row_indices, causal=causal, block_mask_indices=base_blockmask)

    # Forward pass
    #print("pt0")
    fwd_time_ms = do_bench(flash_block_attention_call)
    # paddle._funcpaddle.config.donated_buffer=False
    # Backward pass
    #print("pt1")
    block_out = flash_block_attention_call()
    bwd_time_ms = do_bench(lambda: block_out.backward(gradOut, retain_graph=True))
    #print("pt2")
    total_time_ms = fwd_time_ms + bwd_time_ms

    density = 1 - real_sparsity
    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd')
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd')
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd')

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, real_sparsity



def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, sparsity, causal=False, device="cuda"):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = paddle.zeros(1, nrow, ncol, device=device, dtype=paddle.bool)
    total_block_num = 0

    density = 1.0 - sparsity
    if not density == 0.0 and not density == 1.0:
        for i in range(nrow): # do in reverse order
            idx = nrow - i - 1
            if causal:
                available_col_num = max(0, ncol - i)
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, paddle.randperm(available_col_num)[:num_one]] = True
            else:
                available_col_num = ncol
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, paddle.randperm(available_col_num)[:num_one]] = True
    elif density == 1.0:
        base_mask[0] = paddle.ones_like(base_mask[0])
        total_block_num = nrow * ncol
    else:
        total_block_num = nrow * ncol
    
    calculated_block_num = base_mask.sum().item()
    real_sparsity = 1.0 - calculated_block_num / total_block_num
    return base_mask, real_sparsity

block_size = 128

def get_sparsity_list(sampling_steps, seqlen, causal):
    blockmask_element_num = (seqlen // block_size) ** 2 // (2 if causal else 1)
    stride = max(blockmask_element_num // sampling_steps, 1)
    actual_steps = (blockmask_element_num + stride - 1) // stride
    sparsity_list = []
    for i in range(actual_steps):
        sparse_rate = (1 + i * stride) / blockmask_element_num
        if sparse_rate > 0.95 or sparse_rate < 0.0:
            continue
        sparsity_list.append(sparse_rate)
    return sparsity_list


def main():
    """Run the benchmark with the given examples.

    Args:
        examples: List of examples to run. If "all" is specified, all examples will be run.
    """
    repeats = 15
    block_sparse_repeats = 3
    device = 'cuda:0'
    dtype = 'bfloat16'
    causal = True
    batch_size = 1
    sparsity_sampling_steps = 5
    # seqlen_vals = [1024,2048,4096,8192,16384,32768,65536,65536 * 2]
    seqlen_vals = [65536 * 2]
    headdim = 128
    dim = 4096
    dropout_p = 0.0

    
    all_results = {}
    for seqlen in seqlen_vals:
        results = []
        sparsity_list = get_sparsity_list(sparsity_sampling_steps, seqlen, causal)
        print(f"sparsity_list: {sparsity_list}")
        for causal in [True, False]:
            for sparsity in sparsity_list:
                tmp_results = []
                sum_sparsity, sum_speed, sum_latency = 0, 0, 0
                print(f"seqlen: {seqlen}, causal: {causal}, sparsity: {sparsity}")
                for _ in range(block_sparse_repeats):        
                    fw_time, bw_time, total_time, fw_flops, bw_flops, total_flops, fw_tflops, bw_tflops, total_tflops, sparsity = test_block_mask(B=batch_size, H=dim // headdim, S=seqlen, D=headdim, dtype=dtype, skip_correctness=True, print_mask=False, device=device, causal=causal,sparsity=sparsity)
                    tmp_results.append([f"{fw_time:.4f}", f"{bw_time:.4f}", f"{total_time:.4f}", f"{fw_flops:.4f}", f"{bw_flops:.4f}", f"{total_flops:.4f}", f"{fw_tflops:.4f}", f"{bw_tflops:.4f}", f"{total_tflops:4f}", f"{sparsity:.4f}"])
                tmp_array = np.array(tmp_results).astype(float)
                # print(tmp_array)
                avg_vals = tmp_array.mean(axis=0)
                results.append([causal] + avg_vals.tolist())        
        headers = [
                "Causal",
                "FW Time (ms)",
                "BW Time (ms)",
                "TOTAL Time (ms)",
                "FW FLOPs",
                "BW FLOPs",
                "TOTAL FLOPs",
                "FW TFLOPs/s",
                "BW TFLOPs/s",
                "TOTAL TFLOPs/s",
                "Sparsity",
            ]
        print(results)
        print(
            tabulate(
                results,
                headers=headers,
                tablefmt="grid",
            )
        )
        content2=tabulate(results, headers=headers, tablefmt="tsv")
        print(content2)
        os.makedirs(f"{dtype}", exist_ok=True)
        text_file = open(f"{dtype}/flashblockattention_{batch_size}_{seqlen}_{dim // headdim}_{headdim}.csv","w")
        text_file.write(content2)
        text_file.close()
    
if __name__ == "__main__":
    paddle.set_flags({'FLAGS_flash_attn_version': 3})
    main()

