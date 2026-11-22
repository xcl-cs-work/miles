"""One AC2 step end to end against a fake engine, a fake reward model and a character-level tokenizer."""

import asyncio
from argparse import Namespace
from dataclasses import dataclass, field

import pytest
from examples.ac2 import rollout
from examples.ac2.config import AC2Config
from examples.ac2.rollout import AC2RolloutFn
from tests.fast.examples.ac2.test_ac2_critic import CharTokenizer
from tests.fast.fixtures.args_fixtures import parser_defaults, resolve_parse_boundary_configs

from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.inference_rollout import inference_rollout_common
from miles.rollout.inference_rollout.inference_rollout_common import compute_sampling_params
from miles.utils.types import Sample

GROUP_SIZE = 4
NUM_SLOTS = 2
MAX_RESPONSE_LEN = 64
CHUNK = 8
# a refill response: 30 thinking tokens, then the answer; correct for problem "p0" only
REFILL_RESPONSE = "<think>" + "t" * 23 + "</think>ANS"


class _FakeState:
    def __init__(self, args):
        self.args = args
        self.tokenizer = CharTokenizer()
        self.generate_fn_semaphore = asyncio.Semaphore(64)
        self.sampling_params = compute_sampling_params(
            args, temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=args.rollout_max_response_len
        )


class _FakeDataSource:
    def __init__(self, num_problems: int = 3):
        self.num_problems, self.next = num_problems, 0

    def get_samples(self, count: int) -> list[list[Sample]]:
        groups = []
        for _ in range(count):
            prompt = f"p{self.next % self.num_problems}"
            groups.append([Sample(prompt=prompt, label="ANS", index=self.next) for _ in range(GROUP_SIZE)])
            self.next += 1
        return groups


@dataclass
class _FakeEngine:
    """Continuations generate ``continuation_len`` tokens; critic calls answer ``critic_reply``."""

    continuation_len: int = 12
    critic_reply: str = "0.5"
    critic_contexts: list[str] = field(default_factory=list)
    generations: list[dict] = field(default_factory=list)

    async def __call__(self, url, payload, headers=None):
        text_in = CharTokenizer().decode(payload["input_ids"])
        if text_in.endswith("Q value: "):
            self.critic_contexts.append(text_in)
            return _output(self.critic_reply, finish="stop")
        self.generations.append(payload)
        cap = payload["sampling_params"]["max_new_tokens"]
        n = min(cap, self.continuation_len)
        return _output("c" * n, finish="length" if n == cap else "stop")


def _output(text: str, *, finish: str) -> dict:
    return {
        "text": text,
        "meta_info": {"finish_reason": {"type": finish}, "output_token_logprobs": [(-0.5, ord(c)) for c in text]},
    }


async def _fake_generate_and_rm(state, sample, sampling_params, evaluation=False):
    prompt_ids = rollout.compute_prompt_ids_from_sample(state, sample)
    sample.tokens = prompt_ids + state.tokenizer.encode(REFILL_RESPONSE)
    sample.response, sample.response_length = REFILL_RESPONSE, len(REFILL_RESPONSE)
    sample.status = Sample.Status.COMPLETED
    sample.reward = 1.0 if sample.prompt == "p0" else 0.0
    return sample


async def _fake_rm(args, samples, inplace_set_reward_field=False):
    for sample in samples:
        sample.reward = 1.0 if sample.response.endswith("ANS") else 0.25


def _args(**overrides) -> Namespace:
    ac2_defaults = {name: info.default for name, info in AC2Config.model_fields.items()}
    args = Namespace(
        **{
            **parser_defaults(),
            **ac2_defaults,
            "rollout_batch_size": NUM_SLOTS,
            "n_samples_per_prompt": GROUP_SIZE,
            "rollout_max_response_len": MAX_RESPONSE_LEN,
            "advantage_estimator": "grpo",
            "grpo_std_normalization": False,
            "aux_sft_lr": 1e-6,
            "ac2_chunk_tokens": CHUNK,
            "ac2_num_refill": 3,
            "sglang_router_ip": "engine",
            "sglang_router_port": 1,
            **overrides,
        }
    )
    return resolve_parse_boundary_configs(args)


@pytest.fixture
def engine(monkeypatch) -> _FakeEngine:
    fake = _FakeEngine()
    monkeypatch.setattr(inference_rollout_common, "GenerateState", _FakeState)
    monkeypatch.setattr(rollout, "post", fake)
    monkeypatch.setattr(rollout, "generate_and_rm", _fake_generate_and_rm)
    monkeypatch.setattr(rollout, "batched_async_rm", _fake_rm)
    monkeypatch.setattr(rollout, "compute_routing_headers", lambda args, sample: {})
    monkeypatch.setattr(rollout, "router_routes_by_key", lambda args: False)
    monkeypatch.setattr(
        rollout, "compute_prompt_ids_from_sample", lambda state, sample: state.tokenizer.encode(f"[{sample.prompt}]")
    )
    return fake


def _make_fn(**overrides) -> AC2RolloutFn:
    return AC2RolloutFn(RolloutFnConstructorInput(args=_args(**overrides), data_source=_FakeDataSource()))


async def _step(fn: AC2RolloutFn, rollout_id: int = 0):
    return await fn(RolloutFnTrainInput(rollout_id=rollout_id))


def _flat(output) -> list[Sample]:
    return [sample for group in output.samples for sample in group]


class TestColdStart:
    async def test_fresh_problems_run_to_completion_from_the_empty_prefix(self, engine):
        fn = _make_fn()
        output = await _step(fn)

        assert [len(group) for group in output.samples] == [GROUP_SIZE] * NUM_SLOTS
        for group in output.samples:
            assert len({sample.group_index for sample in group}) == 1
            for sample in group:
                assert sample.metadata["ac2_route"] == "full" and sample.metadata["ac2_prefix_len"] == 0
                assert sample.response_length == engine.continuation_len
                assert sample.reward == 0.25 and not sample.remove_sample
        assert len({sample.index for sample in _flat(output)}) == NUM_SLOTS * GROUP_SIZE

    async def test_refill_fills_the_replay_buffer_and_the_bank(self, engine):
        fn = _make_fn()
        await _step(fn)
        assert len(fn.ac2.replay) == 3
        passed = [entry for entry in fn.ac2.replay.entries if entry.passed]
        assert [entry.reference for entry in passed] == ["ANS"]
        assert list(fn.ac2.bank.values()) == ["ANS"]

    async def test_every_prefix_yields_a_target_and_a_critic_row(self, engine):
        fn = _make_fn()
        output = await _step(fn)
        assert len(engine.critic_contexts) == NUM_SLOTS  # one probe per prefix, nothing chunked yet
        assert [record.target for record in fn.ac2.critic_records.records] == [0.2, 0.2]  # round(0.25), ties down
        assert len(output.aux_sft_rows) == NUM_SLOTS
        for row in output.aux_sft_rows:
            assert row.weight == 1.0 and row.tokens[-row.response_length :] == [ord("0"), ord("."), ord("2"), 1]
        assert output.metrics["ac2/critic_targets"] == NUM_SLOTS


class TestReplayedPrefixes:
    async def test_prefixes_are_cut_from_buffered_rollouts_and_stay_context(self, engine):
        fn = _make_fn()
        await _step(fn)
        output = await _step(fn, rollout_id=1)
        for sample in _flat(output):
            prefix_len = sample.metadata["ac2_prefix_len"]
            assert prefix_len % CHUNK == 0 and prefix_len <= 0.9 * len(REFILL_RESPONSE)
            assert sample.response.startswith(REFILL_RESPONSE[:prefix_len])
            assert len(sample.tokens) - sample.response_length == len(f"[{sample.prompt}]") + prefix_len

    async def test_ready_problems_stop_at_the_chunk_and_the_critic_values_them(self, engine):
        fn = _make_fn(ac2_audit_fraction=0.0)
        await _step(fn)
        fn.ac2.readiness.ready.update(entry.qid for entry in fn.ac2.replay.entries)
        output = await _step(fn, rollout_id=1)
        for sample in _flat(output):
            assert sample.metadata["ac2_route"] == "short" and sample.metadata["ac2_endpoint"] == "critic"
            assert sample.response_length == CHUNK and sample.reward == 0.5
        assert output.metrics["ac2/critic_valued"] == NUM_SLOTS * GROUP_SIZE
        assert all(payload["sampling_params"]["max_new_tokens"] == CHUNK for payload in engine.generations[-8:])

    async def test_audited_ready_problems_still_run_to_completion(self, engine):
        fn = _make_fn(ac2_audit_fraction=1.0)
        await _step(fn)
        fn.ac2.readiness.ready.update(entry.qid for entry in fn.ac2.replay.entries)
        output = await _step(fn, rollout_id=1)
        assert {sample.metadata["ac2_route"] for sample in _flat(output)} == {"audit"}
        assert {sample.metadata["ac2_endpoint"] for sample in _flat(output)} == {"reward"}

    async def test_invalid_critic_values_take_the_group_mean_and_leave_the_loss(self, engine):
        engine.critic_reply = "maybe"
        fn = _make_fn(ac2_audit_fraction=0.0)
        await _step(fn)
        fn.ac2.readiness.ready.update(entry.qid for entry in fn.ac2.replay.entries)
        output = await _step(fn, rollout_id=1)
        assert all(sample.remove_sample and sample.reward == 0.0 for sample in _flat(output))
        assert output.metrics["ac2/critic_invalid"] == NUM_SLOTS * GROUP_SIZE


async def test_a_record_with_a_reference_trains_both_prompts_at_half_weight(engine):
    fn = _make_fn()
    await _step(fn)
    for record in fn.ac2.critic_records.records:
        record.reference = "ANS"
    rows = fn._critic_rows(fn.ac2.critic_records.records[0])
    assert [row.weight for row in rows] == [0.5, 0.5]
    with_reference, without = (CharTokenizer().decode(row.tokens) for row in rows)
    assert "reference correct solution here:\n\nANS" in with_reference and "reference" not in without


async def test_state_survives_save_and_load(engine, tmp_path):
    fn = _make_fn()
    await _step(fn)
    fn.save(tmp_path)
    restored = _make_fn()
    restored.load(tmp_path)
    assert len(restored.ac2.replay) == 3 and restored.ac2.bank == fn.ac2.bank
    assert restored.ac2.next_seq == fn.ac2.next_seq


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"grpo_std_normalization": True}, "--disable-grpo-std-normalization"),
        ({"aux_sft_lr": None}, "--aux-sft-lr"),
        ({"n_samples_per_prompt": 1}, "at least 2"),
        ({"ac2_cut_high": 1.0}, "--ac2-cut-high"),
    ],
)
def test_an_unsupported_configuration_fails_at_construction(engine, overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_fn(**overrides)
