---
title: Qwen3 MoE
description: Launch Qwen3-30B-A3B full-parameter and LoRA recipes, and multi-node Qwen3-235B-A22B.
---
## 1. Model Introduction

[Qwen3 MoE](https://github.com/QwenLM/Qwen3) is the Mixture-of-Experts branch of the Qwen3 series, available in two sizes: 30 B-A3B (single-node) and 235 B-A22B (multi-node).

**Key highlights:**

- **Sparse MoE architecture**: 30 B / 3 B-active and 235 B / 22 B-active variants, scaling capacity without proportional compute cost.
- **Strong reasoning and coding**: shares the Qwen3 generation's improvements in instruction following, math, and tool usage.
- **Long-context capability**: 256 K-token context inherited from the Qwen3 series.
- **Flexible scaling**: 30 B fits a single 8-GPU node; 235 B is the canonical multi-node target with FP8 rollout.

## 2. Supported Variants

| Model | Active / Total | HF ID |
|---|---|---|
| Qwen3-30B-A3B | 3 B / 30 B | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| Qwen3-235B-A22B | 22 B / 235 B | [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) |

## 3. Environment Setup

### 3.1 Required env vars

The 235 B launcher requires:

```bash
export MASTER_ADDR=<head node IP>
```

Everything else is a Typer flag: `--model-dir` (default `/root/models`) for the checkpoints, `--data-dir` (default `/root/datasets`) for the datasets, `--output-dir` (default `/root/shared_data`) for what the run writes. Point them at a shared FS path reachable from every node. The 30 B launcher reads no env vars at all.

### 3.2 Download model + datasets

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir /root/models/Qwen3-30B-A3B
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024     --local-dir /root/datasets/aime-2024

# 235 B (multi-node, FP8 by default)
hf download Qwen/Qwen3-235B-A22B-FP8 --local-dir /root/models/Qwen3-235B-A22B-FP8
```

### 3.3 HF → Megatron `torch_dist` conversion

```bash
MODEL_ARGS_LINE="$(python3 miles/utils/external_utils/model_args_utils.py qwen3-30B-A3B)" || exit 1
read -ra MODEL_ARGS <<< "${MODEL_ARGS_LINE}"
PYTHONPATH=/root/Megatron-LM torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/models/Qwen3-30B-A3B \
   --save          /root/models/Qwen3-30B-A3B_torch_dist
```

Drive the conversion across more GPUs / nodes for the 235 B variant; the launcher reads `<model-dir>/Qwen3-235B-A22B_torch_dist` as `--ref-load`.

## 4. Launch

### 4.1 Quick start

```bash
# 30 B (1 node × 8 GPU) — launcher handles download + conversion + submit
cd /root/miles
python scripts/run_qwen3_30b_a3b.py

# 235 B (8 actor nodes × 8 GPU + a 64-GPU rollout pool)
export MASTER_ADDR=...
python scripts/run_qwen3_235b_a22b.py
```

### 4.2 Multi-node fan-out

`run_qwen3_235b_a22b.py` ssh-fans-out to workers via `--ray-hostfile` (default `/root/mpi_rack_hostfile`) itself; you only need `MASTER_ADDR` set on the head node. Pass `--no-join-ray-workers` when the cluster is already joined. The full-parameter `run_qwen3_30b_a3b.py` launcher is single-node.

### 4.3 Qwen3-30B-A3B LoRA on two single-GPU nodes

`scripts/run_qwen3_30b_a3b_lora.py` trains attention LoRA on one H100 80 GB and
serves rollouts on another. It reuses the native Qwen3 GRPO recipe, loads the BF16
Hugging Face checkpoint directly, and broadcasts adapter tensors through NCCL.
Routed experts remain frozen. This layout needs two allocated GPUs, one on each
node, with the same Miles environment and checkpoint paths available to both.

Use the [installation guide](/getting-started/installation) for the patched
training and serving dependencies. Download the checkpoint and GSM8K once into
shared storage:

```bash
python scripts/run_qwen3_30b_a3b_lora.py prepare \
  --model-dir /shared/models --data-dir /shared/datasets
```

For interactive testing, keep SGLang in its own allocation step and attach
trainer jobs to it. This retains the frozen base when a trainer exits or fails.
Activate the same environment **inside each scheduler allocation**. To require
InfiniBand for adapter tensors, configure NCCL on both nodes before starting
either process (replace `mlx5_0:1` with the active local HCA and port):

```bash
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA='=mlx5_0:1'
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
```

Each node must have working verbs/provider libraries. `NCCL_NET=IB` makes a
missing IB transport fail instead of silently using sockets. Connection setup
and control RPCs still use IP networking. Confirm `Using network IB` and
`via NET/IB` in the actual weight-update connection logs; GPU-direct versus
host-staged RDMA depends on the installed drivers.

```bash
# Training node, with 12 allocated CPUs and one GPU:
ray start --head --node-ip-address "$TRAIN_IP" --port 6379 \
  --num-gpus 1 --num-cpus 12 --dashboard-host 127.0.0.1 \
  --min-worker-port 0 --max-worker-port 0 --disable-usage-stats

# Inference node: keep this process and its allocation alive between trainer runs.
python scripts/run_qwen3_30b_a3b_lora.py serve \
  --model-dir /shared/models --engine-host "$ROLLOUT_IP" --engine-port 30000
```

For mixed-precision serving, `serve --hf-checkpoint /shared/models/Qwen3-30B-A3B-FP8-experts`
can use a separately prepared checkpoint of the same base, while `train` keeps
the original BF16 checkpoint. Optional `--sglang-quantization` and
`--sglang-moe-runner-backend` flags select serving overrides; otherwise SGLang
reads quantization from the checkpoint and chooses its backend. These options
require compatible checkpoint metadata and kernels on the inference GPU. For
selective expert FP8, preserve the attention projections used by LoRA, routers,
norms, embeddings, and output head in their original precision. The launcher
keeps the model dtype BF16; it does not quantize weights or validate the derived
checkpoint's quality. Validate memory use and accuracy on the selected backend
before a training run.

The inference node needs no Ray worker. Preserve the scheduler's
`CUDA_VISIBLE_DEVICES`, and ensure `ray status` advertises only the training
GPU: an external engine's GPU must not also be available to Ray for training.
Once SGLang is healthy, submit from the training allocation shell:

```bash
MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR="$TRAIN_IP" \
  RAY_ADDRESS=http://127.0.0.1:8265 \
  python scripts/run_qwen3_30b_a3b_lora.py train \
    --model-dir /shared/models --data-dir /shared/datasets \
    --output-dir /shared/runs --megatron-path /shared/Megatron-LM \
    --rollout-external-engine-addrs "$ROLLOUT_IP:30000" \
    --num-rollout 2 --rollout-max-response-len 128 --no-enable-wandb
```

For each trainer retry, use the same external address. Miles reconnects the
NCCL weight-update group and publishes the trainer's adapter before sampling;
it does not own or stop the standalone SGLang process. If a trainer fails during
a weight update, clear that incomplete session before retrying:

```bash
curl --fail --noproxy '*' -H 'Content-Type: application/json' \
  -d '{"abort":true}' "http://$ROLLOUT_IP:30000/end_weight_update"
curl --fail --noproxy '*' -H 'Content-Type: application/json' \
  -d '{}' "http://$ROLLOUT_IP:30000/continue_generation"
```

An initialization failure before weight synchronization needs no session reset.
See [Disaggregated Rollout](/advanced/disaggregated-rollout) for external-engine
ownership and compatibility requirements.

This smoke invocation caps each completion at 128 tokens. For a training run,
increase the rollout count and response length. The launcher's dataclass owns
the memory, batch, adapter and transport defaults; `--help` lists their overrides.
It defaults to graceful cleanup, which together with external Ray preserves the
existing cluster and processes. The single-GPU memory settings are a starting
point and require validation on the installed runtime.

## 5. Recipe Configuration

### 5.1 Parallelism

| Launcher | Backend | TP | PP | CP | EP | expert-TP | `max_tokens_per_gpu` | Actor GPUs |
|---|---|---|---|---|---|---|---|---|
| `run_qwen3_30b_a3b.py` (H100, 1 node) | Megatron | 4 | 1 | 1 | 8 | 1 | 32768 | 8 (1 × 8) |
| `run_qwen3_235b_a22b.py` | Megatron | 4 | 4 | 2 | 16 | 1 | 16384 | 64 (8 × 8) |
| `run_qwen3_sft.py --model-name Qwen3-235B-A22B` | Megatron | 4 | 1 | 1 | 32 | 1 | 9216 | 32 (4 × 8) |

`run_qwen3_235b_a22b.py` sets `--decoder-last-pipeline-num-layers 22` to balance the layer count across PP=4. Its rollout is disaggregated, so it needs a further 64 GPUs (`--rollout-num-gpus`) on top of the actor pool.

### 5.2 Algorithm

- **30 B launcher**: GRPO with `--eps-clip 0.2 --eps-clip-high 0.28`.
- **235 B launcher**: GSPO (`--advantage-estimator gspo`, `--eps-clip 4e-4`); `--use-kl-loss` is not passed.

### 5.3 Rollout & SGLang

`run_qwen3_30b_a3b.py` (H100, 1 node, BF16 rollout):

```bash
--rollout-num-gpus-per-engine 8
--sglang-mem-fraction-static 0.7
--sglang-cuda-graph-max-bs-decode 512
```

`run_qwen3_235b_a22b.py`:

```bash
--rollout-num-gpus-per-engine 32
--sglang-mem-fraction-static 0.7
--sglang-enable-dp-attention
--sglang-dp-size 4
--sglang-ep-size 32
--sglang-enable-dp-lm-head
--sglang-cuda-graph-bs-decode 1 2 4 8 16 24 ... 256
--sglang-moe-a2a-backend deepep
--sglang-deepep-mode auto
```

### 5.4 Optimizer

Both `run_qwen3_30b_a3b.py` (H100, 1 node) and `run_qwen3_235b_a22b.py` enable CPU Adam:

```bash
--optimizer-cpu-offload
--overlap-cpu-optimizer-d2h-h2d
--use-precision-aware-optimizer
```

`run_qwen3_30b_a3b.py` removes them when running on Blackwell (`B200/B300/GB200/GB300`) per the hardware match in the launcher.

### 5.5 Notable quirks

- **30 B launcher** supports FP8 / MXFP8 / INT4 rollout, Blackwell hardware, Megatron-bridge mode, and MIS via Typer flags.
- **235 B defaults to the FP8 HF checkpoint** — pass `--no-rollout-fp8` to roll out from the BF16 directory instead.
- **R3 not on by default**; opt-in via `run_qwen3_30b_a3b.py --enable-mis` (TIS / RS) for routing-stability experiments.

## 6. Pairs Well With

- [Low Precision RL](/advanced/low-precision)
- [Rollout Routing Replay (R3)](/advanced/miles-router)
