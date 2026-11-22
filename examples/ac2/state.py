"""AC2's state across steps: the replay buffer, the critic's records, its readiness per problem, and references.

Token sequences are held as int32 arrays: at paper scale the buffers hold thousands of prefixes of up to 50k tokens.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from miles.utils.types import Sample


def as_token_array(ids) -> np.ndarray:
    return np.asarray(ids, dtype=np.int32)


@dataclass
class Trajectory:
    """A finished rollout of a fresh problem; AC2 cuts prefixes from it."""

    qid: str
    template: Sample  # the problem as the data source gave it, before generation
    prompt_ids: np.ndarray
    response_ids: np.ndarray
    passed: bool
    reference: str | None  # its final solution, kept only when it passed
    seq: int


class ReplayBuffer:
    """FIFO of finished trajectories; prefixes are drawn uniformly over problems, then over a problem's entries."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: list[Trajectory] = []

    def __len__(self) -> int:
        return len(self.entries)

    def admit(self, trajectories: list[Trajectory]) -> None:
        self.entries.extend(trajectories)
        del self.entries[: max(0, len(self.entries) - self.capacity)]

    def draw(self, count: int, rng: np.random.Generator) -> list[Trajectory]:
        """``count`` entries: a shuffled pass over the distinct problems, wrapping when there are fewer."""
        by_qid: dict[str, list[Trajectory]] = {}
        for entry in self.entries:
            by_qid.setdefault(entry.qid, []).append(entry)
        qids = sorted(by_qid)
        order = rng.permutation(len(qids))
        drawn = []
        for i in range(count):
            candidates = by_qid[qids[order[i % len(qids)]]]
            drawn.append(candidates[int(rng.integers(len(candidates)))])
        return drawn


def choose_cut(num_tokens: int, rng: np.random.Generator, *, low: float, high: float, grain: int) -> int:
    """A cut on the ``grain`` lattice inside ``[low, high] * num_tokens``, leaving at least one token to generate."""
    lo = math.floor(low * num_tokens)
    hi = max(lo, math.floor(high * num_tokens))
    k_lo, k_hi = -(-lo // grain), hi // grain
    cut = int(rng.integers(k_lo, k_hi + 1)) * grain if k_lo <= k_hi else lo
    return max(0, min(cut, num_tokens - 1))


@dataclass
class CriticRecord:
    """A trained critic target: the state (problem plus prefix) and the grid-rounded mean of its endpoint values."""

    qid: str
    prompt_ids: np.ndarray
    prefix_ids: np.ndarray
    reference: str | None
    target: float


class CriticBuffer:
    """FIFO of critic records; each step trains on a draw without replacement."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.records: list[CriticRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def extend(self, records: list[CriticRecord]) -> None:
        self.records.extend(records)
        del self.records[: max(0, len(self.records) - self.capacity)]

    def draw(self, count: int, rng: np.random.Generator) -> list[CriticRecord]:
        order = rng.permutation(len(self.records))
        return [self.records[i] for i in order[:count]]


@dataclass(frozen=True)
class ProbeObservation:
    """The critic's prediction V(s) at a prefix, against that prefix's target; ``error`` 1 for an invalid one."""

    qid: str
    prediction: float | None
    error: float


@dataclass
class Readiness:
    """Which problems the critic is trusted on; a problem stays ready once it turns ready.

    A problem turns ready on a step whose pooled mean error over the last ``window`` steps is below
    ``threshold_global`` when its own prefix error that step is below ``threshold_local``, the critic has
    predicted a nonzero value on it before, and (optionally) the reference bank holds a solution of it.
    """

    threshold_global: float
    threshold_local: float
    window: int
    require_nonzero: bool
    require_reference: bool
    ready: set[str] = field(default_factory=set)
    nonzero_seen: set[str] = field(default_factory=set)
    errors: deque = field(default_factory=deque)

    def is_ready(self, qid: str) -> bool:
        return qid in self.ready

    @property
    def window_mean_error(self) -> float | None:
        pooled = [e for step in self.errors for e in step]
        return sum(pooled) / len(pooled) if pooled and len(self.errors) == self.window else None

    def observe(self, probes: list[ProbeObservation], *, bank: dict[str, str]) -> list[str]:
        """Add this step's probe errors and return the problems that turned ready."""
        for probe in probes:
            if probe.prediction is not None and probe.prediction > 1e-9:
                self.nonzero_seen.add(probe.qid)
        self.errors.append([probe.error for probe in probes])
        while len(self.errors) > self.window:
            self.errors.popleft()
        mean_error = self.window_mean_error
        if mean_error is None or mean_error >= self.threshold_global:
            return []
        turned = []
        for probe in probes:
            if probe.prediction is None or probe.error >= self.threshold_local or probe.qid in self.ready:
                continue
            if self.require_nonzero and probe.qid not in self.nonzero_seen:
                continue
            if self.require_reference and probe.qid not in bank:
                continue
            self.ready.add(probe.qid)
            turned.append(probe.qid)
        return turned


@dataclass
class AC2State:
    """Everything AC2 carries from step to step, saved and loaded with the rollout function."""

    replay: ReplayBuffer
    critic_records: CriticBuffer
    readiness: Readiness
    bank: dict[str, str] = field(default_factory=dict)  # one reference solution per problem, never replaced
    next_seq: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def take_seq(self, count: int = 1) -> int:
        """The first of ``count`` fresh sequence numbers (sample and group indices)."""
        first = self.next_seq
        self.next_seq += count
        return first
