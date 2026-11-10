import os
import shlex
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)

from miles.utils.external_utils.command_utils.base_backend import BaseCommandBackend

_SCRIPT = REPO_ROOT / "scripts/run_gpt_oss_20b_lora_disaggregated.py"


@pytest.fixture
def launcher(monkeypatch):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    return import_launch_script(_SCRIPT), recording


def _train_tokens(recording):
    submissions = [command for command in recording.commands if "ray job submit" in command]
    assert len(submissions) == 1
    return shlex.split(submissions[0])


def _value(tokens, flag):
    return tokens[tokens.index(flag) + 1]


def test_training_and_rollout_use_separate_single_gpu_roles(launcher, tmp_path):
    module, recording = launcher
    call_entrypoint(module, "train", {}, sandbox=tmp_path)

    tokens = _train_tokens(recording)
    assert _value(tokens, "--actor-num-nodes") == "1"
    assert _value(tokens, "--actor-num-gpus-per-node") == "1"
    assert _value(tokens, "--num-gpus-per-node") == "2"
    assert _value(tokens, "--rollout-num-gpus") == "1"
    assert _value(tokens, "--rollout-num-gpus-per-engine") == "1"
    for flag in (
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--context-parallel-size",
        "--expert-model-parallel-size",
        "--expert-tensor-parallel-size",
    ):
        assert _value(tokens, flag) == "1"
    assert {"--colocate", "--sequence-parallel", "--optimizer-cpu-offload"}.isdisjoint(tokens)
    assert _value(tokens, "--update-weight-transfer-mode") == "broadcast"
    assert _value(tokens, "--megatron-to-hf-mode") == "bridge"
    assert _value(tokens, "--sglang-dtype") == "bfloat16"
    assert _value(tokens, "--lora-type") == "canonical_lora"
    assert _value(tokens, "--target-modules") == "gate_up_proj,down_proj"
    assert _value(tokens, "--sglang-lora-backend") == "triton"
    assert "--no-sglang-lora-use-virtual-experts" in tokens


def test_slurm_gpu_visibility_survives_ray_start_and_cleanup(launcher, monkeypatch, tmp_path):
    module, recording = launcher
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,6")
    exec_command_cpu = BaseCommandBackend.exec_command_cpu
    visible_devices = []

    def record_visible_devices(self, command, **kwargs):
        visible_devices.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return exec_command_cpu(self, command, **kwargs)

    monkeypatch.setattr(BaseCommandBackend, "exec_command_cpu", record_visible_devices)
    call_entrypoint(module, "train", {}, sandbox=tmp_path)

    assert visible_devices and set(visible_devices) == {"3,6"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,6"
    assert any("ray start --head" in command and "--num-gpus 2" in command for command in recording.commands)
    assert any(command.startswith("[launcher lifetime] ray job submit") for command in recording.commands)
    assert recording.commands[0] == recording.commands[-1] == "ray stop --grace-period 60"
    assert not any("pkill" in command or "unset CUDA_VISIBLE_DEVICES" in command for command in recording.commands)


@pytest.mark.parametrize(
    "overrides",
    [{"num_nodes": 2}, {"num_gpus_per_node": 1}, {"num_gpus_per_node": 4}],
)
def test_an_allocation_without_the_required_two_gpu_split_is_rejected(launcher, overrides):
    module, recording = launcher

    with pytest.raises(ValueError, match="one node with two GPUs"):
        module.ScriptArgs(**overrides)

    assert not recording.commands


def test_preparation_and_training_keep_paths_with_spaces_intact(launcher, tmp_path):
    module, recording = launcher
    overrides = {
        "model_dir": str(tmp_path / "model files"),
        "data_dir": str(tmp_path / "training data"),
        "output_dir": str(tmp_path / "adapter outputs"),
        "run_id": "quoted-path-run",
    }
    call_entrypoint(module, "full_train", overrides, sandbox=tmp_path)

    downloads = [shlex.split(command) for command in recording.commands if command.startswith("hf download")]
    assert [_value(tokens, "--local-dir") for tokens in downloads] == [
        str(Path(overrides["model_dir"]) / "gpt-oss-20b-bf16"),
        str(Path(overrides["data_dir"]) / "dapo-math-17k"),
    ]
    tokens = _train_tokens(recording)
    assert _value(tokens, "--hf-checkpoint") == str(Path(overrides["model_dir"]) / "gpt-oss-20b-bf16")
    assert _value(tokens, "--prompt-data") == str(Path(overrides["data_dir"]) / "dapo-math-17k/dapo-math-17k.jsonl")
    assert _value(tokens, "--save") == str(Path(overrides["output_dir"]) / overrides["run_id"] / "checkpoints")
