"""GPT-OSS-20B expert LoRA GRPO with one H100 trainer and one H100 rollout GPU.

Run inside an isolated Miles container on one node with two 80 GB H100s. The
launcher preserves the allocation's CUDA_VISIBLE_DEVICES, starts Ray, and loads
the BF16 checkpoint directly through Megatron-Bridge. No conversion is needed.
The split topology is an adaptation of the maintained colocated LoRA recipe;
it still needs validation on the target GPUs.

Args:
  --num-nodes / --num-gpus-per-node: Fixed to one node and two GPUs.
  --model-dir / --data-dir / --output-dir: Input and adapter checkpoint directories.
  --megatron-path: Megatron-LM checkout installed in the container.
  --num-rollout: Number of GRPO iterations (default: 2).
  --max-seq-len / --rollout-max-response-len: Total and response token limits.
  --sglang-mem-fraction-static: Rollout GPU memory fraction (default: 0.8).

Examples:
  python scripts/run_gpt_oss_20b_lora_disaggregated.py prepare
  python scripts/run_gpt_oss_20b_lora_disaggregated.py train
  python scripts/run_gpt_oss_20b_lora_disaggregated.py full-train --num-rollout 10
"""

import shlex
from dataclasses import dataclass, field
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = "lmsys/gpt-oss-20b-bf16"
_MODEL_NAME = "gpt-oss-20b-bf16"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    num_gpus_per_node: int = 2
    graceful_ray_stop: bool = True
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"

    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    num_rollout: int = 2
    rollout_batch_size: int = 2
    n_samples_per_prompt: int = 4
    global_batch_size: int = 8
    rollout_max_response_len: int = 1024
    max_seq_len: int = 2048
    lr: float = 1e-5
    sglang_mem_fraction_static: float = 0.8
    sglang_max_running_requests: int = 4
    save: bool = True
    save_interval: int = 1
    enable_wandb: bool = True
    extra_args: str = ""

    def __post_init__(self) -> None:
        if self.num_nodes != 1 or self.num_gpus_per_node != 2:
            raise ValueError("This recipe requires one node with two GPUs: one trainer and one rollout GPU.")
        if not 0 < self.rollout_max_response_len < self.max_seq_len:
            raise ValueError("--max-seq-len must exceed a positive --rollout-max-response-len.")
        if not 0 < self.sglang_mem_fraction_static < 1:
            raise ValueError("--sglang-mem-fraction-static must be between zero and one.")
        if self.sglang_max_running_requests < 1:
            raise ValueError("--sglang-max-running-requests must be positive.")


def _download_inputs(args: ScriptArgs) -> None:
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p -- {shlex.quote(args.model_dir)} {shlex.quote(args.data_dir)}")
    backend.exec_command_cpu(
        f"hf download {_HF_REPO} --local-dir {shlex.quote(str(Path(args.model_dir) / _MODEL_NAME))}"
    )
    backend.exec_command_cpu(
        "hf download --repo-type dataset zhuzilin/dapo-math-17k "
        f"--local-dir {shlex.quote(str(Path(args.data_dir) / 'dapo-math-17k'))}"
    )


def _execute(args: ScriptArgs) -> None:
    ckpt_args = (
        f"--hf-checkpoint {shlex.quote(str(Path(args.model_dir) / _MODEL_NAME))} " "--megatron-to-hf-mode bridge "
    )
    if args.save:
        ckpt_args += (
            f"--save {shlex.quote(str(Path(args.output_dir) / args.run_id / 'checkpoints'))} "
            f"--save-interval {args.save_interval} "
        )

    lora_args = (
        f"--lora-rank {args.lora_rank} --lora-alpha {args.lora_alpha} --lora-dropout {args.lora_dropout} "
        "--lora-type canonical_lora --target-modules gate_up_proj,down_proj "
        "--no-sglang-lora-use-virtual-experts "
    )
    rollout_args = (
        f"--prompt-data {shlex.quote(str(Path(args.data_dir) / 'dapo-math-17k/dapo-math-17k.jsonl'))} "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle --rm-type math "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} --global-batch-size {args.global_batch_size} "
        f"--rollout-max-response-len {args.rollout_max_response_len} --rollout-temperature 1 "
        f"--rollout-max-prompt-len {args.max_seq_len - args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
    )
    perf_args = (
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--micro-batch-size 1 --seq-length {args.max_seq_len} "
    )
    optimizer_args = (
        f"--optimizer adam --lr {args.lr} --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
    )
    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    sglang_args = (
        "--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --sglang-dtype bfloat16 "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        f"--sglang-context-length {args.max_seq_len} "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
        f"--sglang-cuda-graph-max-bs-decode {args.sglang_max_running_requests} "
        f"--sglang-max-lora-rank {args.lora_rank} "
        "--sglang-lora-backend triton --sglang-moe-runner-backend triton --sglang-decode-log-interval 1000 "
    )
    misc_args = (
        "--train-backend megatron --attention-dropout 0.0 --hidden-dropout 0.0 "
        "--qkv-format bshd --attention-backend auto --update-weight-transfer-mode broadcast "
        "--update-weight-buffer-size 536870912 --actor-num-nodes 1 --actor-num-gpus-per-node 1 "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id) if args.enable_wandb else ""
    args.create_backend().execute_train(
        train_args=" ".join(
            block.strip()
            for block in (
                ckpt_args,
                lora_args,
                rollout_args,
                perf_args,
                optimizer_args,
                grpo_args,
                sglang_args,
                misc_args,
                wandb_args,
                args.extra_args,
            )
            if block.strip()
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="gpt-oss-20b",
        megatron_path=args.megatron_path,
        job_lifetime="launcher",
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the BF16 checkpoint and training prompts."""
    _download_inputs(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Train an expert LoRA adapter using prepared inputs."""
    _execute(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    """Download inputs and train the adapter."""
    _download_inputs(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
