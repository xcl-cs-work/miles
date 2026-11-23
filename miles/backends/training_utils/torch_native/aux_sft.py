"""The auxiliary SFT objective (``--aux-sft-lr``).

The rollout function returns supervised rows beside the policy batch (``RolloutFnTrainOutput.aux_sft_rows``).
Each rollout trains them once, after one policy optimizer step, with a second optimizer state over the same
weights: its own AdamW moments, learning rate and gradient clip. AC2's generative critic (``examples/ac2``) is
the user: the policy model is also the critic, and the critic's NLL must not share the policy's Adam moments.
"""

import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.data.rollout import DataIterator, count_microbatches
from miles.backends.training_utils.loss.hub.losses import sft_loss_function
from miles.backends.training_utils.metrics.log_utils import log_train_step
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.replay import routing_replay
from miles.backends.training_utils.torch_native.step_runner import StepRunner
from miles.platforms import current_platform
from miles.ray.rollout.train_data_conversion import AUX_SFT_KEYS
from miles.utils.ft_utils.process_group_utils import GeneralPGUtil
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions

# multimodal_train_inputs stays None: the forward reads it, and aux rows carry no media
AUX_SFT_TRAIN_KEYS = [
    "tokens",
    "loss_masks",
    "total_lengths",
    "response_lengths",
    "multimodal_train_inputs",
    "aux_sft_weights",
]

FetchBatches = Callable[[Iterable[int], DataIterator, list[str]], Iterator[dict]]


def take_aux_sft_batch(rollout_data: dict, *, sample_device: torch.device | None) -> dict | None:
    """Pop the aux SFT columns off this rank's shard, as the per-sample fields the trainer batches.

    Every rank holds the same number of rows (see ``_split_aux_sft_by_dp``), so all of them return None together.
    """
    columns = {key: rollout_data.pop(key) for key in AUX_SFT_KEYS if key in rollout_data}
    if not columns or len(columns["aux_sft_tokens"]) == 0:
        return None
    device = sample_device if sample_device is not None else current_platform.current_device()
    tokens = [torch.as_tensor(t, dtype=torch.long, device=device) for t in columns["aux_sft_tokens"]]
    response_lengths = [int(n) for n in columns["aux_sft_response_lengths"]]
    return {
        "tokens": tokens,
        "loss_masks": [torch.ones(n, dtype=torch.int, device=device) for n in response_lengths],
        "total_lengths": [t.numel() for t in tokens],
        "response_lengths": response_lengths,
        "aux_sft_weights": [float(w) for w in columns["aux_sft_weights"]],
    }


@dataclass
class AuxSftLrController:
    """Halves the aux learning rate when the aux step moves the weights much more than the policy steps.

    One update per rollout: a breach is ``aux / policy > ratio``; ``patience`` breaches in a row, or one
    non-finite aux step, halve the rate down to ``min_lr``. A rollout without an aux step resets the streak.
    """

    lr: float
    min_lr: float
    ratio: float
    patience: int
    breach_streak: int = 0

    def update(self, *, aux_displacement: float | None, policy_displacement: float, finite: bool) -> float:
        if not finite:
            self.breach_streak = 0
            self.lr = max(self.min_lr, self.lr * 0.5)
        elif aux_displacement is None:
            self.breach_streak = 0
        else:
            breach = aux_displacement / max(policy_displacement, 1e-12) > self.ratio
            self.breach_streak = self.breach_streak + 1 if breach else 0
            if self.breach_streak >= self.patience:
                self.breach_streak = 0
                self.lr = max(self.min_lr, self.lr * 0.5)
        return self.lr


class AuxSftTrainer:
    """The aux SFT step of one torch-native actor: batch schedule, loss, learning-rate control and logging.

    The backend owns the optimizer and supplies a step runner whose ``apply_step`` clips with
    ``--aux-sft-clip-grad`` and skips the optimizer step on a non-finite gradient norm.
    """

    def __init__(self, args, *, optimizer: torch.optim.Optimizer, parameters: Callable[[], Iterable[torch.Tensor]]):
        self._args = args
        # The targets are scored at the model's own temperature: --rollout-temperature belongs to the policy loss.
        self._loss_args = args.model_copy(update={"rollout_temperature": 1.0})
        self._optimizer = optimizer
        self._parameters = parameters
        self._controller = (
            AuxSftLrController(
                # a resumed optimizer carries the rate an earlier halving left, not --aux-sft-lr
                lr=optimizer.param_groups[0]["lr"],
                min_lr=args.aux_sft_min_lr,
                ratio=args.aux_sft_lr_halving_ratio,
                patience=args.aux_sft_lr_halving_patience,
            )
            if args.aux_sft_lr_halving_ratio is not None
            else None
        )
        self._reset_rollout()

    def _reset_rollout(self) -> None:
        self._policy_displacement = 0.0
        self._aux_displacement: float | None = None
        self._step_log: dict[str, float] | None = None

    def aux_step_index(self, num_policy_steps: int) -> int:
        return min(self._args.aux_sft_after_actor_step, num_policy_steps - 1)

    @contextmanager
    def measure_policy_step(self):
        """Wraps one policy optimizer step to add its displacement to the rollout's total."""
        if self._controller is None:
            yield
            return
        with _measure_displacement(self._parameters()) as result:
            yield
        self._policy_displacement += result[0]

    def train_step(self, batch: dict, *, runner: StepRunner, fetch_batches: FetchBatches) -> None:
        state = get_parallel_state()
        data_iterator, num_microbatches = _build_iterator(self._args, batch)
        normalizer = _dp_sum(sum(batch["aux_sft_weights"]), group=state.intra_dp.group)
        runner.zero_grad()
        # aux rows were never generated, so MoE routing replay has nothing recorded for them
        with routing_replay.stage(routing_replay.FALLTHROUGH):
            logs = runner.forward_backward_step(
                fetch_batches(range(num_microbatches), data_iterator, AUX_SFT_TRAIN_KEYS),
                partial(_aux_sft_loss, self._loss_args, normalizer),
            )
            lr = self._optimizer.param_groups[0]["lr"]
            if self._controller is None:
                metrics = runner.apply_step()
            else:
                with _measure_displacement(self._parameters()) as result:
                    metrics = runner.apply_step()
                self._aux_displacement = result[0]
        runner.zero_grad()
        nll = _dp_sum(float(sum(log["nll"] for log in logs)), group=state.intra_dp.group) / normalizer
        self._step_log = {"loss": nll, "grad_norm": metrics.grad_norm, "lr": lr, "rows": normalizer}

    def end_rollout(self, rollout_id: int) -> None:
        """Log the rollout's aux step and let the controller set the next rollout's learning rate."""
        step_log = self._step_log
        extra = {}
        if self._controller is not None:
            finite = step_log is None or (math.isfinite(step_log["loss"]) and math.isfinite(step_log["grad_norm"]))
            next_lr = self._controller.update(
                aux_displacement=self._aux_displacement if finite else None,
                policy_displacement=self._policy_displacement,
                finite=finite,
            )
            for group in self._optimizer.param_groups:
                group["lr"] = next_lr
            if self._aux_displacement is not None:
                extra["displacement_ratio"] = self._aux_displacement / max(self._policy_displacement, 1e-12)
            extra["next_lr"] = next_lr
        if step_log is not None:
            log_train_step(
                args=self._args,
                loss_dict={"loss": step_log["loss"], "rows": step_log["rows"]},
                grad_norm=step_log["grad_norm"],
                rollout_id=rollout_id,
                step_id=0,
                num_steps_per_rollout=1,
                role="aux_sft",
                extra_metrics={"lr": step_log["lr"], **extra},
                should_log=get_parallel_state().is_metrics_rank,
            )
        self._reset_rollout()


def _build_iterator(args, batch: dict) -> tuple[DataIterator, int]:
    """Seqlen-balanced microbatches; their count is the DP-wide max, so every rank runs as many backward passes."""
    state = get_parallel_state()
    lengths = batch["total_lengths"]
    if args.use_dynamic_batch_size:
        local = count_microbatches(lengths, args.max_tokens_per_gpu * state.cp.size)
    else:
        local = math.ceil(len(lengths) / args.backend.micro_batch_size)
    count = torch.tensor([local], dtype=torch.int, device=current_platform.current_device())
    dp_group = state.effective_dp.group
    GeneralPGUtil.create(dp_group).all_reduce(count, dp_group, op=dist.ReduceOp.MAX)
    # every rank holds the same number of rows, so the max never exceeds the local row count
    partitions = get_seqlen_balanced_partitions(lengths, int(count.item()), equal_size=False)
    return DataIterator(batch, micro_batch_indices=partitions), len(partitions)


def _aux_sft_loss(args, normalizer: float, logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    """The weighted sample-mean NLL of the microbatch's targets, over the DP-total weight of the step."""
    # The reducer divides each row by its token count over its weight; weights are in (0, 1], so the
    # denominator stays at least one token and the reducer's clamp never applies.
    denominators = [
        mask.sum() / weight for mask, weight in zip(batch["loss_masks"], batch["aux_sft_weights"], strict=True)
    ]
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        qkv_format=args.qkv_format,
        max_seq_lens=batch.get("max_seq_lens"),
        denominators=denominators,
    )
    nll, _ = sft_loss_function(args, batch, logits, reducer)
    # the sharded backward averages gradients over DP, which the DP size cancels
    loss = nll / normalizer * get_parallel_state().intra_dp.size
    return loss, {"nll": nll.detach().float().item()}


def _dp_sum(value: float, *, group) -> float:
    total = torch.tensor([value], dtype=torch.float64, device=current_platform.current_device())
    dist.all_reduce(total, group=group)
    return float(total.item())


@contextmanager
def _measure_displacement(parameters: Iterable[torch.Tensor]):
    """Yields a one-slot list that holds the L2 norm of the parameters' change across the block."""
    params = [p for p in parameters if p.requires_grad]
    before = [p.detach().clone() for p in params]
    result: list[float] = []
    yield result
    with torch.no_grad():
        for b, p in zip(before, params, strict=True):
            b.sub_(p.detach())
        norm = torch.nn.utils.get_total_norm(before)
    if isinstance(norm, DTensor):
        norm = norm.full_tensor()
    result.append(float(norm.item()))
