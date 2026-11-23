"""AC2 (Actor-Critic with Action Chunking) on Qwen3-0.6B with the FSDP backend, colocated on one node.

The policy model is also its own critic: the AC2 rollout function (examples/ac2/rollout.py) cuts prefixes from
a replay buffer, values capped chunks with the prompted critic, and returns critic targets that train through the
aux SFT objective (--aux-sft-lr) with a second optimizer state. Hyperparameters follow the paper's Qwen3-4B run
(arXiv:2609.39247, Appendix C), scaled to a 0.6B model, math datasets with 0/1 rewards and shorter responses.

Needs nothing converted: FSDP loads the HF checkpoint as is.

Args:
    --num-gpus-per-node: GPUs on the node, shared by the trainer and the engine.
    --dataset: dapo-math-17k (default) or gsm8k, a quicker sanity check.
    --rollout-batch-size: Prefixes per step; --n-samples-per-prompt continuations each.
    --max-response-len: Response budget per trajectory, prefix plus new tokens.
    --chunk-tokens: New tokens per chunk on ready problems, and the grain of prefix cuts.
    --critic-train-records: Critic targets trained per step (each one or two rows).

Example:
    python examples/ac2/run_qwen3_0_6b_ac2.py --num-gpus-per-node 4
    python examples/ac2/run_qwen3_0_6b_ac2.py --num-gpus-per-node 1 --dataset gsm8k --rollout-batch-size 4 \\
        --n-samples-per-prompt 4 --max-response-len 1024 --chunk-tokens 256 --critic-train-records 8 \\
        --global-batch-size 8 --num-rollout 3
"""

import math
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

HF_REPO = "Qwen/Qwen3-0.6B"
MODEL_NAME = "Qwen3-0.6B"


@dataclass(frozen=True)
class _Dataset:
    hf_repo: str
    path: str  # under --data-dir
    input_key: str
    rm_type: str


_DATASETS: dict[str, _Dataset] = {
    "dapo-math-17k": _Dataset("zhuzilin/dapo-math-17k", "dapo-math-17k/dapo-math-17k.jsonl", "prompt", "deepscaler"),
    "gsm8k": _Dataset("zhuzilin/gsm8k", "gsm8k/train.parquet", "messages", "math"),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    run_id: str = command_utils.create_run_id()
    num_gpus_per_node: int = 4
    num_rollout: int = 200
    dataset: Literal["dapo-math-17k", "gsm8k"] = "dapo-math-17k"
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    max_response_len: int = 8192
    chunk_tokens: int = 2048
    critic_train_records: int = 64
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    extra_args: str = ""

    @property
    def critic_buffer_capacity(self) -> int:
        # the paper keeps 2.5 steps' draws: 1920 records against 768 trained per step
        return math.ceil(2.5 * self.critic_train_records)


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {HF_REPO} --local-dir {args.model_dir}/{MODEL_NAME}")
    U.hf_download_dataset(_DATASETS[args.dataset].hf_repo, data_dir=args.data_dir)


def execute(args: ScriptArgs):
    U = args.create_backend()
    dataset = _DATASETS[args.dataset]
    model_path = f"{args.model_dir}/{MODEL_NAME}"

    ckpt_args = f"--hf-checkpoint {model_path} "

    rollout_args = (
        "--rollout-function-path examples.ac2.rollout.AC2RolloutFn "
        f"--prompt-data {args.data_dir}/{dataset.path} "
        f"--input-key {dataset.input_key} "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--rm-type {dataset.rm_type} "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.max_response_len} "
        "--rollout-temperature 0.8 "
        f"--global-batch-size {args.global_batch_size} "
    )

    # The paper's AC2 settings; the buffer and draw sizes scale with the batch.
    ac2_args = (
        f"--ac2-chunk-tokens {args.chunk_tokens} "
        "--ac2-cut-high 0.9 "
        "--ac2-audit-fraction 0.25 "
        f"--ac2-replay-capacity {8 * args.rollout_batch_size} "
        f"--ac2-critic-train-records {args.critic_train_records} "
        f"--ac2-critic-buffer-capacity {args.critic_buffer_capacity} "
        "--ac2-ready-threshold-global 0.2 "
        "--ac2-ready-threshold-local 0.18 "
    )

    # The critic's optimizer: AdamW without weight decay, its rate halved when it outpaces the policy.
    critic_args = (
        "--aux-sft-lr 2.828e-6 "
        "--aux-sft-min-lr 7.07e-7 "
        "--aux-sft-clip-grad 0.2 "
        "--aux-sft-weight-decay 0.0 "
        "--aux-sft-adam-beta1 0.9 "
        "--aux-sft-adam-beta2 0.999 "
        "--aux-sft-after-actor-step 0 "
        "--aux-sft-lr-halving-ratio 1.414 "
        "--aux-sft-lr-halving-patience 2 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--disable-grpo-std-normalization "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 2e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.01 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.999 "
        "--clip-grad 0.3 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.6 "

    train_backend_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    perf_args = f"--use-dynamic-batch-size --max-tokens-per-gpu {max(9216, args.max_response_len + 2048)} "

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{ac2_args} "
            f"{critic_args} "
            f"{grpo_args} "
            f"{optimizer_args} "
            f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
            f"{sglang_args} "
            f"{train_backend_args} "
            f"{perf_args} "
            f"{misc_args} "
            f"{args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
