import math
from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.torch_native.aux_sft import (
    AuxSftLrController,
    _aux_sft_loss,
    _measure_displacement,
    take_aux_sft_batch,
)
from miles.utils.arguments import _validate_aux_sft_args

CPU = torch.device("cpu")


class TestTakeAuxSftBatch:
    def test_columns_leave_the_shard_as_trainer_fields(self):
        shard = {
            "tokens": [[9]],
            "aux_sft_tokens": [[1, 2, 3, 4], [5, 6, 7]],
            "aux_sft_response_lengths": [2, 1],
            "aux_sft_weights": [0.5, 1.0],
        }
        batch = take_aux_sft_batch(shard, sample_device=CPU)

        assert set(shard) == {"tokens"}
        assert [t.tolist() for t in batch["tokens"]] == [[1, 2, 3, 4], [5, 6, 7]]
        assert [m.tolist() for m in batch["loss_masks"]] == [[1, 1], [1]]
        assert batch["total_lengths"] == [4, 3]
        assert batch["response_lengths"] == [2, 1]
        assert batch["aux_sft_weights"] == [0.5, 1.0]

    def test_an_empty_side_batch_is_none_and_still_leaves_the_shard(self):
        shard = {"aux_sft_tokens": [], "aux_sft_response_lengths": [], "aux_sft_weights": []}
        assert take_aux_sft_batch(shard, sample_device=CPU) is None
        assert shard == {}

    def test_a_shard_without_the_columns_is_none(self):
        assert take_aux_sft_batch({"tokens": [[1]]}, sample_device=CPU) is None


class TestAuxSftLoss:
    def test_rows_are_weighted_token_means_over_the_step_weight(self):
        """Row r adds w_r * mean_t(-log p) and the step divides by its total weight (here 1.5 + 0.5 elsewhere)."""
        make_parallel_state()
        torch.manual_seed(0)
        vocab = 11
        rows = [torch.tensor([1, 4, 2, 7, 3]), torch.tensor([5, 6, 8, 9])]
        response_lengths, weights = [2, 3], [1.0, 0.5]
        logits = torch.randn(1, sum(len(r) for r in rows), vocab)
        batch = {
            "unconcat_tokens": rows,
            "total_lengths": [len(r) for r in rows],
            "response_lengths": response_lengths,
            "loss_masks": [torch.ones(n, dtype=torch.int) for n in response_lengths],
            "aux_sft_weights": weights,
        }
        normalizer = 2.0

        loss, log = _aux_sft_loss(make_args(vocab_size=vocab), normalizer, logits, batch)

        expected, start = 0.0, 0
        for tokens, n, weight in zip(rows, response_lengths, weights, strict=True):
            log_probs = F.log_softmax(logits[0, start : start + len(tokens) - 1], dim=-1)
            targets = tokens[1:]
            nll = -log_probs[torch.arange(len(targets)), targets][-n:]
            expected += weight * nll.mean().item()
            start += len(tokens)
        assert log["nll"] == pytest.approx(expected, rel=1e-5)
        assert loss.item() == pytest.approx(expected / normalizer, rel=1e-5)


class TestAuxSftLrController:
    def _controller(self, **overrides) -> AuxSftLrController:
        return AuxSftLrController(**{"lr": 1.0, "min_lr": 0.3, "ratio": 2.0, "patience": 2, **overrides})

    def test_patience_breaches_in_a_row_halve_the_rate(self):
        controller = self._controller()
        assert controller.update(aux_displacement=3.0, policy_displacement=1.0, finite=True) == 1.0
        assert controller.update(aux_displacement=3.0, policy_displacement=1.0, finite=True) == 0.5
        assert controller.breach_streak == 0

    def test_a_quiet_rollout_resets_the_streak(self):
        controller = self._controller()
        controller.update(aux_displacement=3.0, policy_displacement=1.0, finite=True)
        controller.update(aux_displacement=1.0, policy_displacement=1.0, finite=True)
        assert controller.update(aux_displacement=3.0, policy_displacement=1.0, finite=True) == 1.0

    def test_a_rollout_without_an_aux_step_resets_the_streak(self):
        controller = self._controller()
        controller.update(aux_displacement=3.0, policy_displacement=1.0, finite=True)
        controller.update(aux_displacement=None, policy_displacement=1.0, finite=True)
        assert controller.breach_streak == 0

    def test_a_non_finite_step_halves_at_once_down_to_the_floor(self):
        controller = self._controller()
        assert controller.update(aux_displacement=None, policy_displacement=1.0, finite=False) == 0.5
        assert controller.update(aux_displacement=None, policy_displacement=1.0, finite=False) == 0.3


def test_displacement_is_the_norm_of_the_change():
    params = [torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))]
    with _measure_displacement(params) as result:
        with torch.no_grad():
            params[0].add_(torch.tensor([3.0, 0.0, 0.0]))
            params[1].add_(torch.tensor([0.0, 4.0]))
    assert result == [pytest.approx(5.0)]
    assert math.isclose(params[0][0].item(), 3.0)


class TestAuxSftArgs:
    def _args(self, **overrides) -> Namespace:
        fields = dict(
            aux_sft_lr=1e-6,
            aux_sft_min_lr=0.0,
            aux_sft_after_actor_step=0,
            aux_sft_lr_halving_ratio=None,
            aux_sft_lr_halving_patience=2,
            train_backend="fsdp",
        )
        return Namespace(**{**fields, **overrides})

    def test_off_by_default_whatever_the_backend(self):
        _validate_aux_sft_args(self._args(aux_sft_lr=None, train_backend="megatron"))

    def test_the_second_optimizer_exists_only_on_fsdp(self):
        with pytest.raises(ValueError, match="FSDP backend only"):
            _validate_aux_sft_args(self._args(train_backend="megatron"))

    def test_the_floor_cannot_exceed_the_rate(self):
        with pytest.raises(AssertionError, match="--aux-sft-min-lr"):
            _validate_aux_sft_args(self._args(aux_sft_min_lr=1e-5))
