import os
import numpy as np
from functools import lru_cache
from typing import Optional, List
import random

from block_sparse_attn import (
    block_sparse_attn_func,
    flash_attn_varlen_func,
)

import torch
import torch.nn.functional as F

from tabulate import tabulate

from triton.testing import do_bench



torch.set_default_device("cuda")
torch.manual_seed(0)

np.random.seed(0)
random.seed(0)

torch._dynamo.config.cache_size_limit = 1000



@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device, _compile=True, BLOCK_SIZE=[128, 128])
    return block_mask


def calculate_tflops(flops: float, time_ms: float, multiplier: int) -> float:
    return multiplier * flops * (1e3 / time_ms) / 1e12

def cal_flops(B, H, Sq, Sk, D, mode='fwd',causal=False):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * B * Sq * Sk * H * D
    if(causal):
        f *= 0.5
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def cal_tflops(flops, time_ms):
    return  flops * (1e3 / time_ms) / 1e12

def print_header(text):
    width = 91
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║ {text.center(width - 4)} ║")
    print("╚" + "═" * (width - 2) + "╝")


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
        data_type = torch.bfloat16
    else:
        data_type = torch.float16
        
    seqlen = S
    batch_size = B
    nheads = H
    headdim = D
    dropout_p = 0.0
    shape = (batch_size * seqlen, nheads, headdim)

    q = torch.randn(shape, device=device, dtype=data_type, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=data_type, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=data_type, requires_grad=True)
    gradOut = torch.randn(shape, device=device, dtype=data_type)

    block_attention_call = lambda: flex_attention(*qkv, score_mod=score_mod, block_mask=block_mask)

    cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=device)
    head_mask_type = torch.tensor([1] * nheads, device=device, dtype=torch.int32)
    base_blockmask, real_sparsity = generate_base_sparsity_mask(seqlen, seqlen, block_size, block_size, block_size, sparsity, causal = causal, device=device)
    base_blockmask = base_blockmask.unsqueeze(0).repeat(batch_size, nheads, 1, 1)
    
    block_attention_call = lambda: block_sparse_attn_func(q, k, v, cu_seqlens, cu_seqlens, head_mask_type, None, base_blockmask, seqlen, seqlen, dropout_p, is_causal=causal, exact_streaming=False)

    # Forward pass
    fwd_time_ms = do_bench(block_attention_call)
    # torch._functorch.config.donated_buffer=False
    # Backward pass
    block_out = block_attention_call()
    bwd_time_ms = do_bench(lambda: block_out.backward(gradOut, retain_graph=True))
    total_time_ms = fwd_time_ms + bwd_time_ms

    density = 1 - real_sparsity
    fwd_flops = density * cal_flops(B, H, S, S, D, mode='fwd', causal = causal)
    bwd_flops = density * cal_flops(B, H, S, S, D, mode='bwd', causal = causal)
    total_flops = density * cal_flops(B, H, S, S, D, mode='fwd_bwd', causal = causal)

    fwd_tflops = cal_tflops(fwd_flops, fwd_time_ms)
    bwd_tflops = cal_tflops(bwd_flops, bwd_time_ms)
    total_tflops = cal_tflops(total_flops, total_time_ms)

    return fwd_time_ms, bwd_time_ms, total_time_ms, fwd_flops, bwd_flops, total_flops, fwd_tflops, bwd_tflops, total_tflops, real_sparsity



def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, sparsity, causal=False, device="cuda"):
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(1, nrow, ncol, device=device, dtype=torch.bool)
    total_block_num = 0

    density = 1.0 - sparsity
    if not density == 0.0 and not density == 1.0:
        for i in range(nrow): # do in reverse order
            idx = nrow - i - 1
            if causal:
                available_col_num = max(0, ncol - i)
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
            else:
                available_col_num = ncol
                total_block_num += available_col_num
                num_one = max(1, int(density * available_col_num))
                base_mask[0][idx, torch.randperm(available_col_num)[:num_one]] = True
    elif density == 1.0:
        base_mask[0] = torch.ones_like(base_mask[0])
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
    seqlen_vals = [1024,2048,4096,8192,16384,32768,65536,65536 * 2]
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
        text_file = open(f"{dtype}/blockattention_{batch_size}_{seqlen}_{dim // headdim}_{headdim}.csv","w")
        text_file.write(content2)
        text_file.close()
    
if __name__ == "__main__":
    main()

