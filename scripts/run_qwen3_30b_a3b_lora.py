"""Qwen3-30B-A3B GRPO LoRA with one H100 80GB trainer and a separate rollout GPU.

Start ``serve`` once on the rollout GPU, then point ``train`` at its address with
``--rollout-external-engine-addrs``. The server keeps the frozen base loaded while
trainers reconnect and publish updated adapters. By default both load the BF16
HF base; the trainer uses mbridge through Miles' native path. Serving may use a
separately prepared mixed-precision checkpoint with a compatible quantization
backend, while the trainer keeps its original BF16 base. Both nodes need this
checkout and matching Python environments. Set ``NCCL_NET=IB`` in the
server and trainer Ray environments to require InfiniBand.
Only attention projections have trainable adapters. Check GPU memory use on the
target machines before increasing the conservative sequence and batch limits.

For an existing trainer Ray cluster, set ``MILES_SCRIPT_EXTERNAL_RAY=1``. Without
external engine addresses, training uses managed engines on a two-node Ray
cluster: the lower-IP node trains and the other serves SGLang.

Args:
  --model-dir / --data-dir: HF checkpoint and task dataset directories.
  --torch-dist: Optional trainer checkpoint override (default: the HF checkpoint).
  --output-dir / --save-dir: Output root and optional adapter checkpoint root.
  --num-rollout: Number of rollout/update iterations (default: 10).
  --max-seq-len: Prompt plus completion budget and SGLang context length.
  --engine-host / --engine-port: Standalone server bind address (default: 0.0.0.0:30000).
  --hf-checkpoint: Checkpoint for this invocation; serving and training may use different copies of the same base.
  --sglang-quantization / --sglang-moe-runner-backend: Optional serving overrides; require compatible weights and kernels.
  --rollout-external-engine-addrs: Repeat once per persistent rollout server.
  --extra-args: Training flags appended after the preset's defaults.

Prepare once when model/data directories are shared, otherwise on each node:

  python scripts/run_qwen3_30b_a3b_lora.py prepare --model-dir /data/models --data-dir /data/datasets
  python scripts/run_qwen3_30b_a3b_lora.py serve --model-dir /data/models
  MILES_SCRIPT_EXTERNAL_RAY=1 MASTER_ADDR=<ray-head-ip> \\
    python scripts/run_qwen3_30b_a3b_lora.py train --model-dir /data/models --data-dir /data/datasets \\
      --rollout-external-engine-addrs <rollout-node-ip>:30000
"""

import shlex
from dataclasses import dataclass, replace
from typing import Literal

import typer
from examples.lora import run_qwen3_lora_native as native

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(native.ScriptArgs):
    model_name: Literal["Qwen3-30B-A3B"] = "Qwen3-30B-A3B"
    num_nodes: int = 1
    num_gpus_per_node: int = 1
    actor_num_gpus_per_node: int | None = 1
    rollout_num_gpus: int | None = 1
    rollout_num_gpus_per_engine: int = 1
    tensor_model_parallel_size: int = 1
    colocate: bool = False
    graceful_ray_stop: bool = True

    target_modules: str = "q_proj,k_proj,v_proj,o_proj"
    check_lora_weight_equal: bool = True
    save_dir: str | None = None
    num_rollout: int = 10
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    rollout_max_response_len: int = 1024
    rollout_max_prompt_len: int = 2048
    max_seq_len: int = 4096
    max_tokens_per_gpu: int = 4096
    log_probs_chunk_size: int = 128
    recompute: bool = True

    sglang_mem_fraction_static: float = 0.90
    sglang_max_running_requests: int = 4
    sglang_chunked_prefill_size: int = 1024
    sglang_disable_cuda_graph: bool = True
    sglang_lora_use_virtual_experts: bool = True
    sglang_quantization: str | None = None
    sglang_moe_runner_backend: str | None = None
    update_weight_transfer_mode: Literal["broadcast", "http-lora"] = "broadcast"
    rollout_external_engine_addrs: tuple[str, ...] = ()
    engine_host: str = "0.0.0.0"
    engine_port: int = 30000

    def __post_init__(self):
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.torch_dist is None:
            self.torch_dist = self.hf_checkpoint
        if self.save_dir is None:
            self.save_dir = f"{self.output_dir}/checkpoints"
        super().__post_init__()


def _sglang_runtime_args(args: ScriptArgs, *, prefix: str = "") -> list[str]:
    values: dict[str, int | str] = {
        "context-length": args.max_seq_len,
        "max-running-requests": args.sglang_max_running_requests,
        "chunked-prefill-size": args.sglang_chunked_prefill_size,
    }
    if args.sglang_quantization is not None:
        values["quantization"] = args.sglang_quantization
    if args.sglang_moe_runner_backend is not None:
        values["moe-runner-backend"] = args.sglang_moe_runner_backend
    argv = [token for name, value in values.items() for token in (f"--{prefix}{name}", str(value))]
    if args.sglang_disable_cuda_graph:
        argv.append(f"--{prefix}disable-cuda-graph")
    return argv


def _train(args: ScriptArgs):
    # An HF directory has no Megatron tracker; explicit selection keeps --load
    # from being replaced with the raw-mode fallback's unset --ref-load.
    ckpt_args = "--exit-on-missing-checkpoint "
    perf_args = (
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        f"--seq-length {args.max_seq_len} "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        f"--log-probs-chunk-size {args.log_probs_chunk_size} "
    )
    if args.recompute:
        perf_args += "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    rollout_args = f"--rollout-max-prompt-len {args.rollout_max_prompt_len} --max-seq-len {args.max_seq_len} "
    if args.rollout_external_engine_addrs:
        rollout_args += shlex.join(["--rollout-external-engine-addrs", *args.rollout_external_engine_addrs]) + " "
    sglang_args = shlex.join(_sglang_runtime_args(args, prefix="sglang-")) + " "
    if not args.sglang_lora_use_virtual_experts:
        sglang_args += "--no-sglang-lora-use-virtual-experts "
    sync_args = f"--update-weight-transfer-mode {args.update_weight_transfer_mode} "
    if args.update_weight_transfer_mode == "http-lora":
        sync_args += "--http-lora-ship tensors "
    # User flags retain the shared launcher's last-argument-wins override order.
    native._train(
        replace(args, extra_args=f"{ckpt_args}{perf_args}{rollout_args}{sglang_args}{sync_args}{args.extra_args}")
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    """Download the frozen HF checkpoint and task dataset without allocating a GPU."""
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {shlex.quote(args.model_dir)} {shlex.quote(args.data_dir)}")
    backend.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {shlex.quote(args.hf_checkpoint)}")
    native._download_dataset(args)


@app.command()
@U.dataclass_cli
def serve(args: ScriptArgs):
    """Keep one SGLang server running independently of trainer jobs."""
    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.hf_checkpoint,
        "--trust-remote-code",
        "--host",
        args.engine_host,
        "--port",
        str(args.engine_port),
        "--tp-size",
        str(args.rollout_num_gpus_per_engine),
        "--dtype",
        "bfloat16",
        "--mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--enable-lora",
        "--lora-backend",
        args.sglang_lora_backend,
        "--max-lora-rank",
        str(args.lora_rank),
        "--max-loras-per-batch",
        "1",
        "--lora-target-modules",
        *args.target_modules.split(","),
        "--skip-server-warmup",
        "--enable-metrics",
        *_sglang_runtime_args(args),
    ]
    # Miles enables this by default for managed engines; attention-only LoRA leaves it inert.
    if args.sglang_lora_use_virtual_experts:
        argv.append("--lora-use-virtual-experts")
    args.create_backend().exec_command_gpu(shlex.join(argv), num_gpus_per_node=args.rollout_num_gpus_per_engine)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Submit GRPO and synchronize adapters to external or managed rollout engines."""
    _train(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
