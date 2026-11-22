"""AC2 (Actor-Critic with Action Chunking; "Trust the Critic More", arXiv:2609.39247) as a miles rollout function.

Each training step:

1. Refill: ``--ac2-num-refill`` fresh problems run to completion into a FIFO replay buffer; they enter no loss.
2. Prefixes: ``--rollout-batch-size`` buffered trajectories are drawn (uniform over problems) and cut at a random
   prefix s on the ``--ac2-chunk-tokens`` lattice. The very first step, with an empty buffer, uses fresh problems
   at the empty prefix.
3. Continuations: ``--n-samples-per-prompt`` continuations of each s. Problems the critic is not ready on run to
   completion; on ready ones the continuations stop after ``--ac2-chunk-tokens`` new tokens, except a
   ``--ac2-audit-fraction`` of them that still run to completion.
4. Endpoint values: a chunk that stopped at its cap is valued by the critic, which is the policy model itself
   prompted for its expected credit; every other continuation by the reward model. GRPO without std
   normalization turns them into the advantage A_i = v_i - mean_j v_j of the continuation's new tokens, so the
   prefix is context only.
5. Critic: a prefix with enough valid endpoints yields the target round(mean_j v_j). The critic's error on its
   prefix decides readiness, and a draw of the targets trains the critic through the aux SFT objective
   (``--aux-sft-lr``): an NLL step on the same weights with its own optimizer state.
"""

import asyncio
import hashlib
import json
import math
import pickle
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.ac2.config import AC2Config
from examples.ac2.critic import CriticPrompt, extract_reference, parse_critic_value, round_to_grid
from examples.ac2.state import (
    AC2State,
    CriticBuffer,
    CriticRecord,
    ProbeObservation,
    Readiness,
    ReplayBuffer,
    Trajectory,
    as_token_array,
    choose_cut,
)
from miles.rollout.base_types import (
    AuxSftRow,
    RolloutFnConstructorInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    compute_kv_cache_namespace,
    stamp_kv_cache_namespace,
)
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    router_routes_by_key,
    update_sample_from_response,
)
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn, generate_and_rm
from miles.rollout.rm_hub import batched_async_rm
from miles.utils.http_utils import post
from miles.utils.types import Sample

# The critic answers one grid value: "0.3" plus the end of the turn fits in four tokens.
CRITIC_MAX_NEW_TOKENS = 4
_STATE_FILENAME = "ac2_state.pkl"


@dataclass
class _Slot:
    """One prefix s of a step, and how its continuations run."""

    qid: str
    template: Sample
    prompt_ids: list[int]
    prefix_ids: list[int]
    reference: str | None
    route: str  # "full" (unready), "short" (ready, chunked) or "audit" (ready, run to completion)
    cap: int  # new tokens per continuation
    remaining: int  # the response budget left after the prefix
    group_index: int
    routing_key: str | None

    def consumes_critic(self, sample: Sample) -> bool:
        """Whether a continuation stopped at the chunk cap, short of the budget, so the critic values it."""
        return (
            self.route == "short"
            and self.cap < self.remaining
            and sample.status == Sample.Status.TRUNCATED
            and sample.response_length >= self.cap
        )


@dataclass(frozen=True)
class _CriticAnswer:
    value: float | None  # None: the reply was not a grid value, or the prompt did not fit
    overflow: bool  # the prompt did not fit; such a probe is no readiness observation


class AC2RolloutFn(InferenceRolloutFn):
    config_class = AC2Config

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        args = input.args
        _validate_args(args)
        self.args = args
        self.group_size = args.n_samples_per_prompt
        self.num_refill = args.ac2_num_refill or args.rollout_batch_size
        self.min_valid = args.ac2_critic_min_valid or math.ceil(self.group_size / 2)
        self.critic_prompt = CriticPrompt(
            self.state.tokenizer, think_close=args.ac2_think_close, prefill=args.ac2_critic_prefill
        )
        self.ac2 = AC2State(
            replay=ReplayBuffer(args.ac2_replay_capacity),
            critic_records=CriticBuffer(args.ac2_critic_buffer_capacity),
            readiness=Readiness(
                threshold_global=args.ac2_ready_threshold_global,
                threshold_local=args.ac2_ready_threshold_local,
                window=args.ac2_ready_window,
                require_nonzero=args.ac2_ready_require_nonzero,
                require_reference=args.ac2_ready_require_reference,
            ),
            rng=np.random.default_rng(args.ac2_seed),
        )

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / _STATE_FILENAME, "wb") as f:
            pickle.dump(self.ac2, f)

    def load(self, directory: Path) -> None:
        if (path := directory / _STATE_FILENAME).exists():
            with open(path, "rb") as f:
                self.ac2 = pickle.load(f)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        namespace = compute_kv_cache_namespace(self.args, input)
        fresh = self._fresh_problems(self.num_refill)
        templates = [deepcopy(sample) for sample in fresh]
        slots = self._plan_slots()
        refill, groups = await asyncio.gather(
            self._roll_out_refill(fresh, namespace),
            asyncio.gather(*(self._continue_slot(slot, namespace) for slot in slots)),
        )
        probes = await self._score_endpoints(slots, groups, namespace)
        observations, records = self._critic_targets(slots, groups, probes)
        turned_ready = self.ac2.readiness.observe(observations, bank=self.ac2.bank)
        # after readiness: a reference found this step counts from the next one
        self._add_references(refill + [sample for group in groups for sample in group])
        self.ac2.critic_records.extend(records)
        self.ac2.replay.admit(self._trajectories(refill, templates))
        aux_sft_rows = self._critic_train_rows()
        metrics = self._metrics(slots, groups, refill, records, aux_sft_rows, turned_ready)
        return RolloutFnTrainOutput(samples=groups, metrics=metrics, aux_sft_rows=aux_sft_rows)

    # ---------------------------------------------------------------- planning

    def _fresh_problems(self, count: int) -> list[Sample]:
        # the data source copies each prompt --n-samples-per-prompt times; one copy is the problem
        problems = [group[0] for group in self.data_source.get_samples(count)]
        for sample in problems:
            sample.metadata = {**(sample.metadata or {}), "ac2_qid": _problem_id(sample)}
        return problems

    def _plan_slots(self) -> list[_Slot]:
        state, args = self.ac2, self.args
        if len(state.replay) == 0:
            # cold start: fresh problems at the empty prefix
            sources = []
            for sample in self._fresh_problems(args.rollout_batch_size):
                prompt_ids = compute_prompt_ids_from_sample(self.state, sample)
                sources.append((sample, prompt_ids, [], state.bank.get(sample.metadata["ac2_qid"])))
        else:
            sources = [self._cut(entry) for entry in state.replay.draw(args.rollout_batch_size, state.rng)]
        routes = self._routes([sample.metadata["ac2_qid"] for sample, *_ in sources])
        first_group = state.take_seq(len(sources))
        return [
            self._slot(template, prompt_ids, prefix_ids, reference, route, group_index=first_group + i)
            for i, ((template, prompt_ids, prefix_ids, reference), route) in enumerate(
                zip(sources, routes, strict=True)
            )
        ]

    def _cut(self, entry: Trajectory) -> tuple[Sample, list[int], list[int], str | None]:
        args = self.args
        cut = choose_cut(
            len(entry.response_ids),
            self.ac2.rng,
            low=args.ac2_cut_low,
            high=args.ac2_cut_high,
            grain=args.ac2_chunk_tokens,
        )
        # the trajectory's own solution when it passed, else the bank's
        reference = entry.reference if entry.reference is not None else self.ac2.bank.get(entry.qid)
        return entry.template, entry.prompt_ids.tolist(), entry.response_ids[:cut].tolist(), reference

    def _routes(self, qids: list[str]) -> list[str]:
        ready = [i for i, qid in enumerate(qids) if self.ac2.readiness.is_ready(qid)]
        num_audit = math.ceil(len(ready) * self.args.ac2_audit_fraction)
        audit = set(self.ac2.rng.choice(ready, size=num_audit, replace=False).tolist()) if num_audit else set()
        ready = set(ready)
        return ["full" if i not in ready else "audit" if i in audit else "short" for i in range(len(qids))]

    def _slot(self, template, prompt_ids, prefix_ids, reference, route, *, group_index) -> _Slot:
        remaining = self.args.rollout_max_response_len - len(prefix_ids)
        assert remaining >= 1, f"a prefix of {len(prefix_ids)} tokens leaves no response budget"
        cap = min(self.args.ac2_chunk_tokens, remaining) if route == "short" else remaining
        return _Slot(
            qid=template.metadata["ac2_qid"],
            template=template,
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            reference=reference,
            route=route,
            cap=cap,
            remaining=remaining,
            group_index=group_index,
            # one engine serves a prefix's continuations and critic calls, so they share its cache
            routing_key=str(uuid.uuid4()) if router_routes_by_key(self.args) else None,
        )

    # -------------------------------------------------------------- generation

    async def _roll_out_refill(self, fresh: list[Sample], namespace: str | None) -> list[Sample]:
        stamp_kv_cache_namespace(fresh, namespace=namespace)
        return list(
            await asyncio.gather(
                *(
                    generate_and_rm(self.state, sample, self.state.sampling_params, evaluation=False)
                    for sample in fresh
                )
            )
        )

    async def _continue_slot(self, slot: _Slot, namespace: str | None) -> list[Sample]:
        samples = [self._continuation(slot) for _ in range(self.group_size)]
        await asyncio.gather(*(self._generate(sample, slot.cap, namespace) for sample in samples))
        return samples

    def _continuation(self, slot: _Slot) -> Sample:
        """A pending sample whose prompt side is problem plus prefix: the loss covers only the new tokens."""
        sample = deepcopy(slot.template)
        sample.tokens = slot.prompt_ids + slot.prefix_ids
        # the reward model reads the whole response, prefix included
        sample.response = self.state.tokenizer.decode(
            slot.prefix_ids, skip_special_tokens=self.args.rollout_skip_special_tokens
        )
        sample.response_length = 0
        sample.loss_mask = None
        sample.rollout_log_probs = None
        sample.reward = None
        sample.status = Sample.Status.PENDING
        sample.group_index = slot.group_index
        sample.index = self.ac2.take_seq()
        sample.routing_key = slot.routing_key
        sample.metadata = {**sample.metadata, "ac2_prefix_len": len(slot.prefix_ids), "ac2_route": slot.route}
        return sample

    async def _generate(self, sample: Sample, max_new_tokens: int, namespace: str | None) -> None:
        args = self.args
        payload, _ = compute_request_payload(
            args,
            input_ids=sample.tokens,
            sampling_params={**self.state.sampling_params, "max_new_tokens": max_new_tokens},
            evaluation=False,
            kv_cache_namespace=namespace,
        )
        if payload is None:
            raise ValueError(f"--max-seq-len leaves no room to continue a prefix of {len(sample.tokens)} tokens")
        async with self.state.generate_fn_semaphore:
            output = await post(_generate_url(args), payload, headers=compute_routing_headers(args, sample))
        await update_sample_from_response(args, sample, payload=payload, output=output, tokenizer=self.state.tokenizer)

    async def _query_critic(self, context_ids: list[int], *, routing_key: str | None, namespace) -> _CriticAnswer:
        args = self.args
        limit = args.ac2_critic_max_context_len
        if limit is not None and len(context_ids) + CRITIC_MAX_NEW_TOKENS > limit:
            return _CriticAnswer(value=None, overflow=True)
        sampling_params = {
            **self.state.sampling_params,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": CRITIC_MAX_NEW_TOKENS,
            "skip_special_tokens": True,
        }
        payload, _ = compute_request_payload(
            args, input_ids=context_ids, sampling_params=sampling_params, evaluation=True, kv_cache_namespace=namespace
        )
        if payload is None:
            return _CriticAnswer(value=None, overflow=True)
        headers = compute_routing_headers(args, Sample(routing_key=routing_key))
        async with self.state.generate_fn_semaphore:
            output = await post(_generate_url(args), payload, headers=headers)
        return _CriticAnswer(value=parse_critic_value(output["text"]), overflow=False)

    # ---------------------------------------------------------------- values

    async def _score_endpoints(
        self, slots: list[_Slot], groups: list[list[Sample]], namespace: str | None
    ) -> list[_CriticAnswer]:
        """Set every continuation's endpoint value as its reward; return the critic's probe V(s) of each prefix."""
        judged, consumed = [], []
        for slot, group in zip(slots, groups, strict=True):
            for sample in group:
                if slot.consumes_critic(sample):
                    sample.metadata["ac2_endpoint"] = "critic"
                    consumed.append((slot, sample))
                else:
                    sample.metadata["ac2_endpoint"] = "reward"
                    judged.append(sample)

        def query(slot: _Slot, attempt_ids: list[int]):
            context = self.critic_prompt.context_ids(slot.prompt_ids, attempt_ids, slot.reference)
            return self._query_critic(context, routing_key=slot.routing_key, namespace=namespace)

        _, probes, answers = await asyncio.gather(
            batched_async_rm(self.args, judged, inplace_set_reward_field=True) if judged else asyncio.sleep(0),
            asyncio.gather(*(query(slot, slot.prefix_ids) for slot in slots)),
            asyncio.gather(*(query(slot, sample.tokens[len(slot.prompt_ids) :]) for slot, sample in consumed)),
        )
        for sample in judged:
            _check_reward(sample.reward)
        for (_, sample), answer in zip(consumed, answers, strict=True):
            sample.reward = answer.value
        for group in groups:
            _fill_invalid_values(group)
        return list(probes)

    def _critic_targets(
        self, slots: list[_Slot], groups: list[list[Sample]], probes: list[_CriticAnswer]
    ) -> tuple[list[ProbeObservation], list[CriticRecord]]:
        observations, records = [], []
        for slot, group, probe in zip(slots, groups, probes, strict=True):
            values = [sample.reward for sample in group if not sample.remove_sample]
            if len(values) < self.min_valid:
                continue
            target = round_to_grid(sum(values) / len(values))
            records.append(
                CriticRecord(
                    qid=slot.qid,
                    prompt_ids=as_token_array(slot.prompt_ids),
                    prefix_ids=as_token_array(slot.prefix_ids),
                    reference=slot.reference,
                    target=target,
                )
            )
            if not probe.overflow:
                error = 1.0 if probe.value is None else abs(probe.value - target)
                observations.append(ProbeObservation(qid=slot.qid, prediction=probe.value, error=error))
        return observations, records

    # ------------------------------------------------------------- bookkeeping

    def _add_references(self, samples: list[Sample]) -> None:
        """Add one correct solution per problem the bank lacks, from responses the reward model judged."""
        candidates: dict[str, list[Sample]] = {}
        for sample in samples:
            qid = sample.metadata["ac2_qid"]
            # refill responses carry no endpoint tag: the reward model judged them all
            judged = sample.metadata.get("ac2_endpoint", "reward") == "reward" and sample.reward is not None
            if judged and qid not in self.ac2.bank and sample.reward >= self.args.ac2_pass_threshold:
                candidates.setdefault(qid, []).append(sample)
        for qid, passed in candidates.items():
            chosen = passed[int(self.ac2.rng.integers(len(passed)))]
            if (reference := self._reference_of(chosen)) is not None:
                self.ac2.bank[qid] = reference

    def _trajectories(self, refill: list[Sample], templates: list[Sample]) -> list[Trajectory]:
        trajectories = []
        for sample, template in zip(refill, templates, strict=True):
            if sample.response_length == 0 or sample.status == Sample.Status.ABORTED:
                continue
            _check_reward(sample.reward)
            passed = sample.reward >= self.args.ac2_pass_threshold
            split = len(sample.tokens) - sample.response_length
            trajectories.append(
                Trajectory(
                    qid=sample.metadata["ac2_qid"],
                    template=template,
                    prompt_ids=as_token_array(sample.tokens[:split]),
                    response_ids=as_token_array(sample.tokens[split:]),
                    passed=passed,
                    reference=self._reference_of(sample) if passed else None,
                    seq=self.ac2.take_seq(),
                )
            )
        return trajectories

    def _reference_of(self, sample: Sample) -> str | None:
        return extract_reference(sample.response, think_close=self.args.ac2_think_close) or None

    def _critic_train_rows(self) -> list[AuxSftRow]:
        records = self.ac2.critic_records.draw(self.args.ac2_critic_train_records, self.ac2.rng)
        return [row for record in records for row in self._critic_rows(record)]

    def _critic_rows(self, record: CriticRecord) -> list[AuxSftRow]:
        """A record with a reference trains both prompts at weight 1/2; one that does not fit trains nothing."""
        use_both = record.reference is not None and self.args.ac2_critic_train_no_reference
        references = [record.reference, None] if use_both else [record.reference]
        target = self.critic_prompt.target_ids(record.target)
        contexts = [
            self.critic_prompt.context_ids(record.prompt_ids, record.prefix_ids, reference) for reference in references
        ]
        limit = self.args.ac2_critic_max_context_len
        if limit is not None and any(len(context) + len(target) > limit for context in contexts):
            return []
        weight = 1.0 / len(contexts)
        return [AuxSftRow(tokens=context + target, response_length=len(target), weight=weight) for context in contexts]

    def _metrics(self, slots, groups, refill, records, aux_sft_rows, turned_ready) -> dict[str, float]:
        routes = [slot.route for slot in slots]
        continuations = [sample for group in groups for sample in group]
        metrics = {
            "ac2/ready_problems": len(self.ac2.readiness.ready),
            "ac2/turned_ready": len(turned_ready),
            "ac2/slots_full": routes.count("full"),
            "ac2/slots_short": routes.count("short"),
            "ac2/slots_audit": routes.count("audit"),
            "ac2/critic_valued": sum(s.metadata["ac2_endpoint"] == "critic" for s in continuations),
            "ac2/critic_invalid": sum(s.remove_sample for s in continuations),
            "ac2/critic_targets": len(records),
            "ac2/critic_train_rows": len(aux_sft_rows),
            "ac2/replay_size": len(self.ac2.replay),
            "ac2/bank_size": len(self.ac2.bank),
            "ac2/new_tokens": sum(s.response_length for s in continuations + refill),
        }
        if refill:
            # the trained rewards mix critic values into the endpoints; fresh full rollouts measure the outcome
            metrics["rollout/raw_reward"] = sum(s.reward for s in refill) / len(refill)
        if (mean_error := self.ac2.readiness.window_mean_error) is not None:
            metrics["ac2/critic_window_mean_error"] = mean_error
        return metrics


def _fill_invalid_values(group: list[Sample]) -> None:
    """An invalid critic value takes the mean of the group's valid values (0 if none) and leaves the loss."""
    valid = [sample.reward for sample in group if sample.reward is not None]
    fill = sum(valid) / len(valid) if valid else 0.0
    for sample in group:
        if sample.reward is None:
            sample.reward = fill
            sample.remove_sample = True


def _check_reward(reward) -> None:
    if not isinstance(reward, (int, float)) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"AC2 compares rewards with critic values on [0, 1]; got reward {reward!r}")


def _problem_id(sample: Sample) -> str:
    key = json.dumps([sample.prompt, sample.label], sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()


def _generate_url(args) -> str:
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"


def _validate_args(args) -> None:
    problems = []
    if args.advantage_estimator != "grpo" or not args.rewards_normalization or args.grpo_std_normalization:
        problems.append(
            "the advantage is the group-mean-subtracted endpoint value: pass --advantage-estimator grpo "
            "--disable-grpo-std-normalization and keep rewards normalization on"
        )
    if args.aux_sft_lr is None:
        problems.append("the critic trains through the aux SFT objective: set --aux-sft-lr")
    if args.group_rm or args.reward_key:
        problems.append("endpoint values need one scalar reward per response: drop --group-rm and --reward-key")
    if args.length_penalty_coeff > 0:
        problems.append("--length-penalty-coeff would shape critic values as if they were rewards")
    if args.partial_rollout or args.fully_async:
        problems.append("AC2 runs synchronous steps: drop --partial-rollout and --fully-async")
    if args.custom_generate_function_path or args.dynamic_sampling_filter_path:
        problems.append("AC2 generates its own continuations: drop --custom-generate-function-path and filters")
    if args.n_samples_per_prompt < 2:
        problems.append("the group-mean baseline needs --n-samples-per-prompt of at least 2")
    if not 0.0 <= args.ac2_cut_low <= args.ac2_cut_high < 1.0:
        problems.append("--ac2-cut-low and --ac2-cut-high must satisfy 0 <= low <= high < 1")
    if args.ac2_chunk_tokens < 1:
        problems.append("--ac2-chunk-tokens must be positive")
    if not 0.0 <= args.ac2_audit_fraction <= 1.0:
        problems.append("--ac2-audit-fraction must lie in [0, 1]")
    if problems:
        raise ValueError("AC2 rollout: " + "; ".join(problems))
