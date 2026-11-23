# AC2: Actor-Critic with Action Chunking

AC2 trains on chunks of replayed rollouts and values unfinished chunks with the policy itself, prompted as a critic.

It implements [Trust the Critic More](https://arxiv.org/abs/2609.39247) (Wen, Bailey, Mahankali, Ma), whose
reference code is [WhenWen/AC2](https://github.com/WhenWen/AC2). GRPO waits for every trajectory to finish and
gives all its tokens the same advantage. AC2 instead continues short prefixes of past trajectories and assigns
credit per chunk: where a learned critic is trusted, a chunk of `b` new tokens is scored by the critic at its end
and never runs to completion.

## How it maps onto miles

Everything algorithmic lives in the rollout function, `examples.ac2.rollout.AC2RolloutFn`; its module docstring
lists the five phases of a step. Two miles pieces carry the rest:

* **Advantage.** Each continuation's reward is its endpoint value `v_i`: the reward model's score when it
  finished, the critic's value when it stopped at the chunk cap. `--advantage-estimator grpo
  --disable-grpo-std-normalization` turns them into `A_i = v_i - mean_j v_j` per prefix. The prefix sits on the
  prompt side of the sample, so only the new tokens enter the loss.
* **Critic update.** The critic is the policy model, asked in a new user turn for a value on {0, 0.1, ..., 1}
  after a prefilled `Q value: `. Its targets return as `RolloutFnTrainOutput.aux_sft_rows`, which the FSDP
  trainer fits with an NLL step on the same weights but with a second AdamW state, after policy step
  `--aux-sft-after-actor-step`. The `--aux-sft-*` flags set that optimizer (see `miles/utils/args/configs/algo.py`).

A problem's continuations are chunked only once the critic is **ready** on it: its mean error over the last
`--ac2-ready-window` steps is low, its error on that problem's prefix is low, it has predicted a nonzero value
there, and a correct reference solution exists. Ready problems stay ready, and an `--ac2-audit-fraction` of
them still run to completion so their targets stay grounded in rewards.

## Files

* `rollout.py`: the rollout function and its step.
* `critic.py`: the critic's prompt, value grid and parsing.
* `state.py`: the replay buffer, critic records, readiness and reference bank, saved with the rollout checkpoint.
* `config.py`: the `--ac2-*` flags; batch sizes come from `--rollout-batch-size` (prefixes per step),
  `--n-samples-per-prompt` (continuations per prefix) and `--rollout-max-response-len` (prefix plus new tokens).
* `run_qwen3_0_6b_ac2.py`: Qwen3-0.6B on dapo-math-17k or gsm8k, FSDP, colocated on one node.

## Quick start

```bash
python examples/ac2/run_qwen3_0_6b_ac2.py --num-gpus-per-node 4
```

A one-GPU sanity run on gsm8k with short responses:

```bash
python examples/ac2/run_qwen3_0_6b_ac2.py --num-gpus-per-node 1 --dataset gsm8k --rollout-batch-size 4 \
    --n-samples-per-prompt 4 --max-response-len 1024 --chunk-tokens 256 --critic-train-records 8 \
    --global-batch-size 8 --num-rollout 3
```

## What to watch

* `rollout/raw_reward`: the outcome reward of the step's fresh full rollouts, the on-policy accuracy. The trained
  rewards mix in critic values, so `rollout/raw_reward_filtered` is not an accuracy.
* `ac2/critic_window_mean_error`, `ac2/ready_problems`, `ac2/slots_short`: whether the critic is trusted, and on
  how much of the batch.
* `train/aux_sft-loss`, `train/aux_sft-displacement_ratio`, `train/aux_sft-lr`: the critic's fit, and its step size
  against the policy's; the rate halves after `--aux-sft-lr-halving-patience` rollouts above
  `--aux-sft-lr-halving-ratio`.
* `ac2/new_tokens`: generated tokens per step, the decoding cost AC2 reduces.

## Differences from the paper

* **Rewards.** Any scalar reward on [0, 1] from the miles reward model, instead of a proof judge; the prompt asks
  for credit on a "solution" rather than a proof. A chunk stopped at its cap always goes to the critic: the paper
  also sends one that already holds a complete `<proof>` to the judge.
* **Critic loss.** Each row's NLL is the mean over its target tokens; the paper divides the sum by four.
* **Not ported.** The adaptive entropy clip (the paper's `eps_high` controller), and backends other than FSDP.
* **Interleaving.** The paper runs its critic step between two policy minibatches; set `--global-batch-size` to
  half the step's samples and keep `--aux-sft-after-actor-step 0` for the same order, as the launcher does.
