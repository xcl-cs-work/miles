---
title: GPT-OSS 20B
sidebarTitle: GPT-OSS
description: Launch recipe for OpenAI's GPT-OSS 20B — Megatron BF16 on a single 8-GPU node, loading the MXFP4 HF checkpoint through mbridge.
---
## 1. Model Introduction

[GPT-OSS](https://huggingface.co/openai/gpt-oss-20b) is OpenAI's open-weight language model, designed for reasoning, agentic tasks, and developer use cases. miles supports the 20 B variant.

**Key highlights:**

- **Configurable reasoning effort**: low / medium / high reasoning effort selectable per request.
- **Full chain-of-thought**: the reasoning trace is exposed and trainable.
- **MXFP4 native weights**: the HF checkpoint ships in MXFP4 (post-trained) — the BF16 launcher uses mbridge to load HF directly.
- **Sink attention**: requires `--qkv-format bshd` on the Megatron path, which precludes dynamic batch sizing.

## 2. Supported Variants

| Model | HF ID |
|---|---|
| gpt-oss-20b | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |

## 3. Environment Setup

### 3.1 Download model + datasets

```bash
hf download openai/gpt-oss-20b --local-dir /root/models/gpt-oss-20b
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

### 3.2 HF → Megatron `torch_dist` conversion

The launcher needs no `convert_hf_to_torch_dist.py` step: it loads the HF checkpoint directly via `--megatron-to-hf-mode bridge` (mbridge).

## 4. Launch

### 4.1 Quick start

```bash
# Megatron BF16 (1 node × 8 GPU)
cd /root/miles
python scripts/run_gpt_oss_20b.py
```

The checkpoint is read from `--model-dir` (default `/root/models`) and the dataset from
`--data-dir` (default `/root/datasets`).

### 4.2 LoRA on Slurm: one training GPU and one rollout GPU

`scripts/run_gpt_oss_20b_lora_disaggregated.py` prepares and runs a small GRPO
experiment on one node with two 80 GB H100s. Megatron trains expert LoRA adapters
on one GPU; SGLang serves the current adapter on the other. Adapter updates use
NCCL broadcast. This is a configuration to validate on your cluster, not a
recorded two-H100 benchmark.

Use a Miles runtime containing its compatible Megatron-Bridge, Transformer
Engine and SGLang builds. This recipe downloads `lmsys/gpt-oss-20b-bf16`, not the
MXFP4 checkpoint used above. Its Triton expert-LoRA path requires BF16 weights.
The launcher owns the Ray lifecycle and preserves Slurm's GPU visibility.

The following submission example uses Slurm's Pyxis/Enroot container flags.
Set the account, partition, image and **absolute** host paths for your site.
The checkout, Megatron checkout and storage must be accessible on the compute
node. Create the storage directory before submission; downloads require network
access and space for the BF16 checkpoint (roughly 42 GB) plus saved adapters.

```bash
export SLURM_ACCOUNT=your_account
export SLURM_PARTITION=your_h100_partition
export CONTAINER_IMAGE=/absolute/path/to/miles-runtime.sqsh
export MILES_ROOT=/absolute/path/to/miles
export MEGATRON_ROOT=/absolute/path/to/Megatron-LM
export RUN_STORAGE=/absolute/path/to/persistent/storage
mkdir -p "$RUN_STORAGE"

sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
  --nodes=1 --ntasks-per-node=1 --gpus-per-node=2 --exclusive \
  --cpus-per-task=16 --mem=128G --time=02:00:00 \
  --job-name=gpt-oss-lora --output=slurm-%j.out --export=ALL <<'SBATCH'
#!/bin/bash
set -euo pipefail
srun --nodes=1 --ntasks=1 --gpus-per-node=2 \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$MILES_ROOT:/workspace/miles,$MEGATRON_ROOT:/workspace/Megatron-LM,$RUN_STORAGE:/workspace/storage" \
  --container-workdir=/workspace/miles \
  bash -c 'set -euo pipefail
    export PYTHONPATH="/workspace/miles:/workspace/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"
    export HF_HOME=/workspace/storage/hf-cache
    unset RAY_ADDRESS MILES_SCRIPT_EXTERNAL_RAY
    python3 scripts/run_gpt_oss_20b_lora_disaggregated.py full-train \
      --num-nodes 1 --num-gpus-per-node 2 \
      --megatron-path /workspace/Megatron-LM \
      --model-dir /workspace/storage/models \
      --data-dir /workspace/storage/datasets \
      --output-dir /workspace/storage/runs'
SBATCH
```

`--exclusive` isolates the default Ray ports and Ray cleanup from other jobs;
it reserves the node even when the node has more than two GPUs. Select a
two-GPU node if available. Shared-node execution requires site-provided process
and port isolation before dropping this flag. GPU type labels and container
integration are site-specific; the partition above must provide H100s. Add your
site's QOS or GPU-type constraint if needed. On a site without Pyxis, run the
same Python command through its supported Miles container runtime.

The default run is two rollouts with a 2,048-token sequence cap, microbatch one,
full activation recomputation and four concurrent inference requests. No
reference model or critic is allocated. Adapters are saved under
`<output-dir>/<run-id>/checkpoints`. Increase `--num-rollout` after checking that
both training and inference fit. `--help` lists the batch, length, LoRA and
SGLang memory controls.

Use `prepare` to download ahead of time in the same runtime with the same
storage paths, then replace `full-train` with `train` in the batch job. For an
interactive allocation, run the same `srun` command directly.

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | TP | PP | CP | EP | expert-TP | `micro-batch-size` | GPUs |
|---|---|---|---|---|---|---|---|
| `run_gpt_oss_20b.py` (Megatron) | 8 | 1 | 1 | 8 | 1 | 1 | 8 (1 × 8) |

`--use-dynamic-batch-size` is **not** used on the Megatron BF16 path — the in-source comment explains: `--qkv-format bshd` (required for sink attention with TE) is incompatible with dynamic batch size. Only `--micro-batch-size 1` is set. `--sequence-parallel` is on (required for TP + EP).

### 5.2 Algorithm

The recipe uses GRPO with `--eps-clip 0.2 --eps-clip-high 0.28 --entropy-coef 0.00` and `--rm-type math`. **`--use-kl-loss` is not passed** — bridge mode has no Megatron-format reference checkpoint to supply `--ref-load`, which KL loss would need.

### 5.3 Rollout & SGLang

```bash
--rollout-num-gpus-per-engine 4
--sglang-dtype bfloat16
--sglang-decode-log-interval 1000
--sglang-mem-fraction-static 0.70
```

### 5.4 Optimizer

The launcher enables CPU Adam (`--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer`).

### 5.5 Notable quirks

Attention setup:

```bash
--attention-dropout 0.0
--hidden-dropout 0.0
--qkv-format bshd        # required for TE sink attention (SWA + learnable softmax offset)
--attention-backend fused
```

`--qkv-format bshd` is mandated by the sink-attention pattern; in turn it precludes `--use-dynamic-batch-size`. Don't toggle either flag without the other.

The launcher passes no `--load`, and no `--save` unless you ask for it: `--save` exports the
bf16 HF checkpoint to `<model-dir>/gpt-oss-20b-BF16` every 50 steps, which can then be fed
back in as the `--hf-checkpoint`.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
