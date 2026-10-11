"""Small, offline training + KV-cache check. No tokenizer or dataset required.

python -m scripts.smoke_test --device-type cuda --require-rocm
NANOCHAT_COMPILE=0 python -m scripts.smoke_test --device-type cpu
"""

import argparse

import torch

from nanochat.common import (
    COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, autodetect_device_type,
    compile_enabled, compute_init, compute_cleanup, maybe_compile,
)
from nanochat.engine import KVCache
from nanochat.flash_attention import USE_FA3
from nanochat.gpt import GPT, GPTConfig


def run_smoke_test(device):
    config = GPTConfig(sequence_len=32, vocab_size=256, n_layer=2,
                       n_head=2, n_kv_head=1, n_embd=128, window_pattern="L")
    with torch.device(device):
        model = GPT(config)
        model.init_weights()
    optimizer = model.setup_optimizer()
    train_model = maybe_compile(model, dynamic=False)
    tokens = torch.randint(0, config.vocab_size, (2, 17), device=device)
    inputs, targets = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
    # FP16 needs loss scaling; this also checks that embeddings retain FP32
    # master weights so GradScaler can unscale their gradients.
    scaler = torch.amp.GradScaler(device.type) if COMPUTE_DTYPE == torch.float16 else None
    before = model.lm_head.weight.detach().clone()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = train_model(inputs, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        print(f"step {step + 1}: loss={loss.item():.4f}")
    if torch.equal(before, model.lm_head.weight):
        raise RuntimeError("Optimizer did not update the model")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError("Optimizer produced non-finite parameters")

    model.eval()
    cache = KVCache(batch_size=2, num_heads=config.n_kv_head,
                    seq_len=config.sequence_len, head_dim=config.n_embd // config.n_head,
                    num_layers=config.n_layer, device=device, dtype=COMPUTE_DTYPE)
    with torch.inference_mode():
        reference = model(inputs)
        prefill = model(inputs[:, :8], kv_cache=cache)
        chunk = model(inputs[:, 8:-1], kv_cache=cache)
        decode = model(inputs[:, -1:], kv_cache=cache)
        cached = torch.cat((prefill, chunk, decode), dim=1)
        # Different attention kernels/reduction orders can round differently.
        tolerance = 5e-2 if COMPUTE_DTYPE != torch.float32 else 1e-4
        torch.testing.assert_close(cached, reference, atol=tolerance, rtol=tolerance)
    if cache.get_pos() != inputs.size(1):
        raise RuntimeError("KV cache did not advance correctly")
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print("Smoke test passed: forward/backward, Muon+AdamW, GQA, KV prefill/chunk/decode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--require-rocm", action="store_true",
                        help="fail instead of silently using a CPU or NVIDIA GPU")
    args = parser.parse_args()
    device_type = args.device_type or autodetect_device_type()
    if args.require_rocm and (
        device_type != "cuda" or torch.version.hip is None or not torch.cuda.is_available()
    ):
        parser.error("A ROCm GPU is required. Install the ROCm PyTorch build and check /dev/kfd and /dev/dri permissions.")
    _, _, _, _, device = compute_init(device_type)
    print(f"PyTorch: {torch.__version__} | HIP: {torch.version.hip} | device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name} | architecture: {getattr(props, 'gcnArchName', 'NVIDIA')} | memory: {props.total_memory / 2**30:.1f} GiB")
    print(f"dtype: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
    print(f"Attention: {'FA3' if USE_FA3 else 'SDPA'} | compile: {compile_enabled()}")
    try:
        run_smoke_test(device)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
