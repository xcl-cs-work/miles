import numpy as np
import pytest
from examples.ac2.state import (
    CriticBuffer,
    CriticRecord,
    ProbeObservation,
    Readiness,
    ReplayBuffer,
    Trajectory,
    as_token_array,
    choose_cut,
)

from miles.utils.types import Sample


def _trajectory(qid: str, seq: int) -> Trajectory:
    return Trajectory(
        qid=qid,
        template=Sample(prompt=qid),
        prompt_ids=as_token_array([1]),
        response_ids=as_token_array([2, 3]),
        passed=False,
        reference=None,
        seq=seq,
    )


class TestChooseCut:
    def test_cuts_land_on_the_grain_lattice_inside_the_bounds(self):
        rng = np.random.default_rng(0)
        cuts = {choose_cut(50, rng, low=0.0, high=0.9, grain=10) for _ in range(200)}
        assert cuts == {0, 10, 20, 30, 40}

    def test_a_trajectory_shorter_than_a_grain_is_cut_at_zero(self):
        rng = np.random.default_rng(0)
        assert {choose_cut(9, rng, low=0.0, high=0.9, grain=10) for _ in range(20)} == {0}

    def test_a_cut_always_leaves_a_token_to_generate(self):
        assert choose_cut(10, np.random.default_rng(0), low=1.0, high=1.0, grain=10) == 9

    def test_bounds_with_no_lattice_point_fall_back_to_the_low_bound(self):
        assert choose_cut(100, np.random.default_rng(0), low=0.31, high=0.35, grain=10) == 31


class TestReplayBuffer:
    def test_the_oldest_entries_leave_first(self):
        buffer = ReplayBuffer(capacity=3)
        buffer.admit([_trajectory("a", i) for i in range(5)])
        assert [entry.seq for entry in buffer.entries] == [2, 3, 4]

    def test_a_draw_visits_every_problem_before_repeating_one(self):
        """A problem with many buffered rollouts is drawn no more often than one with a single rollout."""
        buffer = ReplayBuffer(capacity=10)
        buffer.admit([_trajectory("a", i) for i in range(6)] + [_trajectory("b", 6), _trajectory("c", 7)])
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert sorted(entry.qid for entry in buffer.draw(3, rng)) == ["a", "b", "c"]
        assert sorted(entry.qid for entry in buffer.draw(6, rng)) == ["a", "a", "b", "b", "c", "c"]


def test_the_critic_buffer_is_a_fifo_drawn_without_replacement():
    buffer = CriticBuffer(capacity=4)
    buffer.extend(
        [CriticRecord(qid=str(i), prompt_ids=None, prefix_ids=None, reference=None, target=0.0) for i in range(6)]
    )
    assert [record.qid for record in buffer.records] == ["2", "3", "4", "5"]
    drawn = buffer.draw(3, np.random.default_rng(0))
    assert len({record.qid for record in drawn}) == 3
    assert len(buffer.draw(10, np.random.default_rng(0))) == 4


class TestReadiness:
    def _readiness(self, **overrides) -> Readiness:
        fields = dict(
            threshold_global=0.2, threshold_local=0.18, window=2, require_nonzero=True, require_reference=True
        )
        return Readiness(**{**fields, **overrides})

    def _probe(self, qid: str, prediction: float | None, error: float) -> ProbeObservation:
        return ProbeObservation(qid=qid, prediction=prediction, error=error)

    def test_nothing_turns_ready_before_the_window_fills(self):
        readiness = self._readiness()
        assert readiness.observe([self._probe("a", 0.5, 0.0)], bank={"a": "ref"}) == []
        assert readiness.observe([self._probe("a", 0.5, 0.0)], bank={"a": "ref"}) == ["a"]
        assert readiness.is_ready("a")

    def test_a_high_pooled_error_blocks_every_problem(self):
        readiness = self._readiness()
        readiness.observe([self._probe("a", 0.5, 0.0), self._probe("b", None, 1.0)], bank={"a": "ref"})
        assert readiness.observe([self._probe("a", 0.5, 0.0)], bank={"a": "ref"}) == []
        assert readiness.window_mean_error == pytest.approx(1 / 3)

    @pytest.mark.parametrize(
        ("probe", "bank", "reason"),
        [
            (("a", 0.5, 0.18), {"a": "ref"}, "the local error is not below the threshold"),
            (("a", None, 0.0), {"a": "ref"}, "an invalid prediction"),
            (("a", 0.5, 0.0), {}, "no reference solution yet"),
        ],
    )
    def test_a_problem_needs_every_local_condition(self, probe, bank, reason):
        readiness = self._readiness(window=1)
        assert readiness.observe([self._probe(*probe)], bank=bank) == [], reason

    def test_a_problem_needs_a_nonzero_prediction_first(self):
        readiness = self._readiness(window=1, threshold_local=0.5)
        assert readiness.observe([self._probe("a", 0.0, 0.1)], bank={"a": "ref"}) == []
        assert readiness.observe([self._probe("a", 0.1, 0.0)], bank={"a": "ref"}) == ["a"]

    def test_readiness_is_sticky(self):
        readiness = self._readiness(window=1)
        readiness.observe([self._probe("a", 0.5, 0.0)], bank={"a": "ref"})
        assert readiness.observe([self._probe("a", None, 1.0)], bank={"a": "ref"}) == []
        assert readiness.is_ready("a")

    def test_the_optional_conditions_can_be_dropped(self):
        readiness = self._readiness(window=1, require_nonzero=False, require_reference=False)
        assert readiness.observe([self._probe("a", 0.0, 0.0)], bank={}) == ["a"]
