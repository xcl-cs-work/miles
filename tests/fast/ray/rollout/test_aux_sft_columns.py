from argparse import Namespace

import pytest
from tests.fast.train_parallel_config_utils import make_train_parallel_config

from miles.ray.rollout.train_data_conversion import (
    AUX_SFT_KEYS,
    build_aux_sft_columns,
    convert_samples_to_train_data,
    split_train_data_by_dp,
)
from miles.rollout.base_types import AuxSftRow
from miles.utils.types import Sample


def _split_args() -> Namespace:
    return Namespace(balance_data=False, multi_lora=False, global_batch_size=4, enable_sample_ownership_checker=False)


def _policy_data(num_samples: int = 4) -> dict:
    return {
        "tokens": [[1, 2, 3, 4]] * num_samples,
        "loss_masks": [[1, 1]] * num_samples,
        "response_lengths": [2] * num_samples,
        "rewards": [0.0] * num_samples,
        "sample_indices": list(range(num_samples)),
    }


def _rows(lengths: list[int]) -> list[AuxSftRow]:
    return [AuxSftRow(tokens=list(range(n)), response_length=2, weight=0.5 if n % 2 else 1.0) for n in lengths]


class TestBuildAuxSftColumns:
    def test_columns_keep_each_row(self):
        columns = build_aux_sft_columns([AuxSftRow(tokens=[5, 6, 7], response_length=1, weight=0.5)])
        assert columns == {"aux_sft_tokens": [[5, 6, 7]], "aux_sft_response_lengths": [1], "aux_sft_weights": [0.5]}

    @pytest.mark.parametrize("response_length", [0, 3])
    def test_a_row_needs_both_context_and_target(self, response_length):
        with pytest.raises(ValueError, match="context and a target"):
            build_aux_sft_columns([AuxSftRow(tokens=[1, 2, 3], response_length=response_length)])

    @pytest.mark.parametrize("weight", [0.0, 1.5])
    def test_a_weight_outside_zero_one_is_rejected(self, weight):
        """The trainer's reducer clamps a row denominator at one token, which a weight above one would hit."""
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            build_aux_sft_columns([AuxSftRow(tokens=[1, 2, 3], response_length=1, weight=weight)])


class TestConvertCarriesAuxSft:
    def _args(self) -> Namespace:
        return Namespace(
            advantage_estimator="grpo",
            rewards_normalization=False,
            length_penalty_coeff=0.0,
            enable_sample_ownership_checker=False,
            reward_key=None,
            use_output_store=False,
            rollout_top_logprobs_num=0,
            use_rollout_routing_replay=False,
            use_rollout_indexer_replay=False,
            multimodal_token_space_processing=False,
            use_dynamic_global_batch_size=False,
        )

    def _samples(self) -> list[Sample]:
        return [Sample(tokens=[1, 2, 3], response_length=1, reward=1.0, index=i) for i in range(2)]

    def test_the_metadata_columns_reach_the_train_data(self):
        aux = build_aux_sft_columns(_rows([4, 5]))
        train_data = convert_samples_to_train_data(
            self._args(),
            self._samples(),
            metadata={"aux_sft": aux},
            custom_convert_samples_to_train_data_func=None,
            custom_reward_post_process_func=None,
        )
        assert {key: train_data[key] for key in AUX_SFT_KEYS} == aux

    def test_a_custom_convert_would_drop_them_and_raises(self):
        with pytest.raises(ValueError, match="would drop the rollout's aux SFT rows"):
            convert_samples_to_train_data(
                self._args(),
                self._samples(),
                metadata={"aux_sft": build_aux_sft_columns(_rows([4]))},
                custom_convert_samples_to_train_data_func=lambda args, samples: {},
                custom_reward_post_process_func=None,
            )


class TestSplitAuxSftByDp:
    def test_every_rank_gets_as_many_rows_and_the_tail_drops(self):
        data = {**_policy_data(), **build_aux_sft_columns(_rows([3, 9, 4, 8, 5, 7, 6]))}
        shards = split_train_data_by_dp(_split_args(), data, make_train_parallel_config(dp_size=3))

        counts = [len(shard["aux_sft_tokens"]) for shard in shards]
        assert counts == [2, 2, 2]
        kept = sorted(len(tokens) for shard in shards for tokens in shard["aux_sft_tokens"])
        assert kept == [3, 4, 5, 7, 8, 9]  # the trailing row (6 tokens) drops
        for shard in shards:
            for tokens, weight in zip(shard["aux_sft_tokens"], shard["aux_sft_weights"], strict=True):
                assert weight == (0.5 if len(tokens) % 2 else 1.0)

    def test_fewer_rows_than_ranks_leaves_every_rank_empty(self):
        """No rank may train an aux step its peers skip: the sharded backward runs on all of them."""
        data = {**_policy_data(), **build_aux_sft_columns(_rows([3]))}
        shards = split_train_data_by_dp(_split_args(), data, make_train_parallel_config(dp_size=2))
        assert [shard["aux_sft_tokens"] for shard in shards] == [[], []]

    def test_a_batch_without_aux_rows_gets_no_aux_columns(self):
        shards = split_train_data_by_dp(_split_args(), _policy_data(), make_train_parallel_config(dp_size=2))
        assert all(key not in shard for shard in shards for key in AUX_SFT_KEYS)
