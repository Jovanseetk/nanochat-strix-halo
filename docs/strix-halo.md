# Strix Halo: Ryzen AI Max+ 395 on Linux

This port targets the Radeon 8060S integrated GPU (`gfx1151`) through ROCm.
PyTorch calls AMD GPUs `cuda`, so use `--device-type=cuda`, not `rocm` or `hip`.
The NPU is not used. See [PyTorch's HIP semantics](https://docs.pytorch.org/docs/2.9/notes/hip.html).

## Install and check the GPU

Install the AMD driver/ROCm prerequisites for your Linux distribution first.
The pinned baseline here is **ROCm 7.2, PyTorch 2.9.1, Python 3.12, Linux x86_64**.
AMD lists the 395 and Ubuntu 24.04.3 in its [ROCm 7.2 Ryzen support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html).
Its [installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installryz/native_linux/install-pytorch.html)
requires the 6.14-1018 OEM kernel or newer and supplies the exact torch/Triton wheels pinned in `requirements-rocm.txt`.
This is a versioned baseline matching nanochat's existing torch pin, not a claim that ROCm 7.2 is the latest release.

With [uv installed](https://docs.astral.sh/uv/getting-started/installation/), run from this checkout:

```bash
bash runs/setup_strix_halo.sh
source .venv/bin/activate
```

The setup script chooses Python 3.12 explicitly (the repository's generic
`.python-version` is 3.10). It installs the existing locked base dependencies,
skips their CPU torch wheel, and installs hash-checked AMD torch/Triton wheels.
The separate wheel file keeps AMD packages out of the ordinary CPU/NVIDIA
dependency resolution. The setup then runs a tiny
offline GPU check: training forward/backward, Muon and AdamW updates, grouped
query attention, and KV-cache prefill/chunk/decode. It fails if an AMD GPU is not
available. No training data or model download is needed for this check.

To repeat it or install development dependencies:

```bash
python -m scripts.smoke_test --device-type cuda --require-rocm
bash runs/setup_strix_halo.sh --group dev
python -m pytest
```

Use this setup only on Linux x86_64 with Python 3.12; the pinned wheel files are
specific to that platform. Rerun the setup script to refresh the environment.
A plain `uv sync` or `uv run` can replace the AMD torch build; use the activated environment's
`python`, or `uv run --no-sync`, after setup.

## Train a small model

```bash
bash runs/strix_halo.sh
```

This downloads two training shards plus validation data, trains a tokenizer if
missing, then runs 100 steps of a six-layer model. It uses one GPU process, full
attention, 512-token sequences, and a microbatch of four. This is a starter run
to verify the pipeline, not a GPT-2-quality training recipe. No run-time or
throughput estimate is claimed for the 395.

Artifacts live in `~/.cache/nanochat-strix-halo` by default. Set
`NANOCHAT_BASE_DIR` to change that. To run longer:

```bash
NANOCHAT_DATA_SHARDS=8 bash runs/strix_halo.sh --num-iterations=1000
```

Additional arguments are passed to `scripts.base_train` after the defaults.
Increase the depth, context, and microbatch gradually after checking memory and
loss. `--total-batch-size` must be divisible by `--device-batch-size` times
`--max-seq-len` for one GPU. The machine's system RAM is shared with the GPU;
available GPU memory depends on firmware, OS, and other workloads. The smoke
test prints the memory PyTorch sees. Reduce microbatch/context first on OOM.

## Evaluate, fine-tune, and chat

In a new shell, activate the environment and use the same artifact directory:

```bash
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-strix-halo"

# Evaluate the base model (downloads evaluation resources on first use).
python -m scripts.base_eval --model-tag=strix-halo --device-batch-size=1 --split-tokens=8192 --max-per-task=16

# A short SFT run (downloads the chat datasets on first use).
python -m scripts.chat_sft --model-tag=strix-halo --num-iterations=100 \
  --device-batch-size=4 --total-batch-size=8192 --eval-tokens=8192 --chatcore-every=-1

python -m scripts.chat_cli --model-tag=strix-halo
```

A short base/SFT run mainly checks that everything executes; useful chat quality
requires much more training. The original `runs/speedrun.sh` still targets eight
NVIDIA H100s and should not be used on this machine.

## Runtime choices and troubleshooting

- ROCm selects BF16 when PyTorch reports support, otherwise FP32. AMD's 7.2
  Ryzen matrix officially validates FP16 only; BF16 on this port must be checked
  with the smoke test on your system. You can use `NANOCHAT_DTYPE=float32` for a
  conservative fallback or `NANOCHAT_DTYPE=float16` for pretraining/SFT, which
  use GradScaler. RL does not currently support FP16 loss scaling.
- Attention uses PyTorch SDPA. NVIDIA FA3 kernels are never loaded on ROCm.
  Prefer `--window-pattern=L`; sliding-window masks can use much more memory.
- Strix Halo defaults to eager execution for both the model and optimizer.
  After the baseline passes, try `NANOCHAT_COMPILE=1 python -m scripts.smoke_test
  --device-type cuda --require-rocm`, then apply that setting to training.
  `NANOCHAT_COMPILE=0` disables compilation everywhere. Set it before launching
  Python, since optimizer decorators are evaluated during import.
- If desired, test AMD's opt-in AOTriton attention kernels by setting
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` before Python starts, as described
  in AMD's installation guide. They are not required by the default setup.
- `--fp8` is rejected on ROCm: this repository's FP8 recipe targets NVIDIA
  Hopper or newer. Do not copy the H100 speedrun's FP8 flag.
- If `torch.cuda.is_available()` is false, confirm `torch.version.hip` is set,
  check `python -m torch.utils.collect_env`, and verify access to `/dev/kfd`
  and `/dev/dri/render*` using your distribution's AMD driver instructions.
- An “invalid device function” / “no kernel image” error usually indicates a
  wheel or library without your GPU target. Confirm the smoke test reports
  `gfx1151` and rerun `bash runs/setup_strix_halo.sh`. Do not set
  `HSA_OVERRIDE_GFX_VERSION` to impersonate another GPU for this native build.
- MFU/MBU can report zero because this repository has no verified 8060S peak
  performance entry. Tokens/sec and measured GPU memory are still useful.

The portability tests and CPU smoke checks can run on other machines, but they
do not establish GPU correctness or performance. Run the ROCm smoke test on
the actual 395 before starting a long training job.
