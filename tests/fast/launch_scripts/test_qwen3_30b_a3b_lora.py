import shlex
from argparse import ArgumentParser, Namespace

import pytest
from sglang.srt.server_args import ServerArgs

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

from miles.backends.megatron_utils.megatron_config import resolve_args_checkpoint_load


def _record(monkeypatch, tmp_path, *, entrypoint="train", external_ray=True, **overrides):
    freeze_environment(monkeypatch)
    if external_ray:
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_qwen3_30b_a3b_lora.py")
    call_entrypoint(module, entrypoint, {"enable_wandb": False, **overrides}, sandbox=tmp_path)
    return recording.commands


def _train_tokens(commands):
    submit = next(command for command in commands if "ray job submit" in command)
    tokens = shlex.split(submit)
    return tokens[tokens.index("--") + 1 :]


def _value(tokens, flag):
    # Launch recipes deliberately append user overrides after their defaults.
    return [tokens[index + 1] for index, token in enumerate(tokens) if token == flag][-1]


def test_separate_one_gpu_nodes_load_the_same_frozen_base_and_sync_only_the_adapter(monkeypatch, tmp_path):
    tokens = _train_tokens(_record(monkeypatch, tmp_path, model_dir="/models", output_dir="/outputs"))

    assert _value(tokens, "--actor-num-nodes") == "1"
    assert _value(tokens, "--actor-num-gpus-per-node") == "1"
    assert _value(tokens, "--num-gpus-per-node") == "1"
    assert _value(tokens, "--rollout-num-gpus") == "1"
    assert _value(tokens, "--rollout-num-gpus-per-engine") == "1"
    assert "--colocate" not in tokens
    assert "--sequence-parallel" not in tokens
    assert _value(tokens, "--tensor-model-parallel-size") == "1"
    assert _value(tokens, "--load") == _value(tokens, "--hf-checkpoint") == "/models/Qwen3-30B-A3B"
    assert _value(tokens, "--megatron-to-hf-mode") == "raw"
    assert _value(tokens, "--target-modules") == "q_proj,k_proj,v_proj,o_proj"
    assert _value(tokens, "--update-weight-transfer-mode") == "broadcast"
    assert "--http-lora-ship" not in tokens
    assert "--rollout-external-engine-addrs" not in tokens
    assert "--check-lora-weight-equal" in tokens
    assert "--use-kl-loss" not in tokens


def test_external_ray_launch_preserves_processes_and_existing_cluster(monkeypatch, tmp_path):
    commands = _record(monkeypatch, tmp_path)

    assert any("ray job submit" in command for command in commands)
    assert all(
        "pkill" not in command and "ray stop" not in command and "ray start" not in command for command in commands
    )


def test_external_engine_addresses_reuse_the_servers_without_process_cleanup(monkeypatch, tmp_path):
    addresses = ("http://rollout-a:30000", "rollout-b:30001")
    commands = _record(monkeypatch, tmp_path, rollout_external_engine_addrs=addresses)
    tokens = _train_tokens(commands)
    start = tokens.index("--rollout-external-engine-addrs") + 1

    assert tokens[start : start + len(addresses)] == list(addresses)
    assert _value(tokens, "--update-weight-transfer-mode") == "broadcast"
    assert all("pkill" not in command and "ray stop" not in command for command in commands)


@pytest.mark.parametrize("virtual_experts", [True, False])
def test_serve_launches_an_independent_lora_engine_with_the_training_limits(monkeypatch, tmp_path, virtual_experts):
    overrides = dict(
        hf_checkpoint="/models/Qwen3 checkpoint",
        engine_host="127.0.0.1",
        engine_port=31000,
        lora_rank=8,
        max_seq_len=3072,
        sglang_mem_fraction_static=0.85,
        sglang_max_running_requests=2,
        sglang_chunked_prefill_size=512,
        sglang_lora_use_virtual_experts=virtual_experts,
    )
    commands = _record(monkeypatch, tmp_path, entrypoint="serve", **overrides)
    assert len(commands) == 1
    tokens = shlex.split(commands[0])
    assert tokens[:3] == ["python3", "-m", "sglang.launch_server"]
    assert "--gated-launch-port" not in tokens
    assert all("ray " not in command and "pkill" not in command for command in commands)

    parser = ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server = parser.parse_args(tokens[3:])
    assert server.model_path == overrides["hf_checkpoint"]
    assert (server.host, server.port) == ("127.0.0.1", 31000)
    assert server.enable_lora
    assert server.max_lora_rank == 8
    assert server.max_loras_per_batch == 1
    assert server.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert server.lora_use_virtual_experts == virtual_experts
    assert server.dtype == "bfloat16"
    assert server.tp_size == 1
    assert server.mem_fraction_static == 0.85
    assert server.disable_cuda_graph

    training = _train_tokens(_record(monkeypatch, tmp_path, **overrides))
    assert _value(training, "--load") == _value(training, "--hf-checkpoint") == server.model_path
    assert str(server.context_length) == _value(training, "--sglang-context-length") == "3072"
    assert _value(training, "--seq-length") == str(server.context_length)
    assert str(server.max_running_requests) == _value(training, "--sglang-max-running-requests") == "2"
    assert str(server.chunked_prefill_size) == _value(training, "--sglang-chunked-prefill-size") == "512"
    assert ("--no-sglang-lora-use-virtual-experts" in training) == (not virtual_experts)


def test_fresh_raw_run_keeps_the_hf_checkpoint_through_load_resolution(monkeypatch, tmp_path):
    checkpoint = tmp_path / "Qwen3-30B-A3B"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    tokens = _train_tokens(_record(monkeypatch, tmp_path, hf_checkpoint=str(checkpoint)))
    args = Namespace(
        load=_value(tokens, "--load"),
        hf_checkpoint=_value(tokens, "--hf-checkpoint"),
        megatron_to_hf_mode=_value(tokens, "--megatron-to-hf-mode"),
        exit_on_missing_checkpoint="--exit-on-missing-checkpoint" in tokens,
        no_load_optim="--no-load-optim" in tokens,
        no_load_rng="--no-load-rng" in tokens,
        finetune="--finetune" in tokens,
        ref_load=None,
        ckpt_step=None,
        ref_ckpt_step=None,
        start_rollout_id=None,
    )

    resume_from_ckpt = resolve_args_checkpoint_load(args, role="actor")

    assert args.load == str(checkpoint)
    assert resume_from_ckpt is False


def test_quantized_serving_reaches_server_args_with_bf16_lora_settings(monkeypatch, tmp_path):
    (command,) = _record(
        monkeypatch,
        tmp_path,
        entrypoint="serve",
        hf_checkpoint="/models/Qwen3-30B-A3B-FP8-experts",
        sglang_quantization="humming",
        sglang_moe_runner_backend="humming",
    )
    parser = ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server = parser.parse_args(shlex.split(command)[3:])

    assert server.model_path == "/models/Qwen3-30B-A3B-FP8-experts"
    assert server.quantization == "humming"
    assert server.moe_runner_backend == "humming"
    assert server.dtype == "bfloat16"
    assert server.enable_lora
    assert server.max_lora_rank == 16
    assert server.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_quantized_rollout_options_do_not_replace_the_trainers_bf16_base(monkeypatch, tmp_path):
    tokens = _train_tokens(
        _record(
            monkeypatch,
            tmp_path,
            hf_checkpoint="/models/Qwen3-30B-A3B",
            rollout_external_engine_addrs=("rollout:30000",),
            sglang_quantization="humming",
            sglang_moe_runner_backend="humming",
        )
    )

    assert _value(tokens, "--load") == _value(tokens, "--hf-checkpoint") == "/models/Qwen3-30B-A3B"
    assert _value(tokens, "--megatron-to-hf-mode") == "raw"
    assert _value(tokens, "--sglang-quantization") == "humming"
    assert _value(tokens, "--sglang-moe-runner-backend") == "humming"
    assert _value(tokens, "--sglang-dtype") == "bfloat16"
    assert _value(tokens, "--lora-rank") == "16"
    assert "--fp8" not in tokens
    assert "--fp8-format" not in tokens


def test_http_adapter_sync_is_an_explicit_option(monkeypatch, tmp_path):
    tokens = _train_tokens(_record(monkeypatch, tmp_path, update_weight_transfer_mode="http-lora"))

    assert _value(tokens, "--update-weight-transfer-mode") == "http-lora"
    assert _value(tokens, "--http-lora-ship") == "tensors"


@pytest.mark.parametrize(("max_seq_len", "max_response_len"), [(3072, 1024), (12288, 8129)])
def test_memory_limits_bound_training_and_serving_on_an_80gb_gpu(
    monkeypatch, tmp_path, max_seq_len, max_response_len
):
    tokens = _train_tokens(
        _record(
            monkeypatch,
            tmp_path,
            max_seq_len=max_seq_len,
            max_tokens_per_gpu=max_seq_len,
            rollout_max_response_len=max_response_len,
        )
    )

    assert _value(tokens, "--recompute-granularity") == "full"
    assert _value(tokens, "--micro-batch-size") == "1"
    assert _value(tokens, "--log-probs-chunk-size") == "128"
    assert _value(tokens, "--seq-length") == _value(tokens, "--max-seq-len") == str(max_seq_len)
    assert _value(tokens, "--sglang-context-length") == _value(tokens, "--max-tokens-per-gpu") == str(max_seq_len)
    assert _value(tokens, "--rollout-max-response-len") == str(max_response_len)
    assert float(_value(tokens, "--sglang-mem-fraction-static")) == 0.9
    assert _value(tokens, "--sglang-max-running-requests") == "4"
    assert "--sglang-disable-cuda-graph" in tokens


def test_user_flags_override_preset_tuning(monkeypatch, tmp_path):
    tokens = _train_tokens(_record(monkeypatch, tmp_path, extra_args="--max-tokens-per-gpu 2048 --seed 17"))

    assert _value(tokens, "--max-tokens-per-gpu") == "2048"
    assert _value(tokens, "--seed") == "17"
    assert tokens.index("--update-weight-transfer-mode") < tokens.index("--seed")


def test_prepare_needs_no_conversion_or_gpu(monkeypatch, tmp_path):
    commands = _record(monkeypatch, tmp_path, entrypoint="prepare", model_dir="/models", data_dir="/datasets")

    assert "hf download Qwen/Qwen3-30B-A3B --local-dir /models/Qwen3-30B-A3B" in commands
    assert any("zhuzilin/gsm8k" in command for command in commands)
    assert all("torchrun" not in command and "ray " not in command for command in commands)


@pytest.mark.parametrize(("colocate", "actor_gpus", "rollout_gpus"), [(True, "8", None), (False, "4", "2")])
def test_shared_native_recipe_preserves_existing_gpu_defaults(
    monkeypatch, tmp_path, colocate, actor_gpus, rollout_gpus
):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "examples/lora/run_qwen3_lora_native.py")
    call_entrypoint(module, "train", {"colocate": colocate, "enable_wandb": False}, sandbox=tmp_path)
    tokens = _train_tokens(recording.commands)

    assert _value(tokens, "--actor-num-gpus-per-node") == actor_gpus
    assert ("--colocate" in tokens) == colocate
    if rollout_gpus is None:
        assert "--rollout-num-gpus" not in tokens
    else:
        assert _value(tokens, "--rollout-num-gpus") == rollout_gpus
