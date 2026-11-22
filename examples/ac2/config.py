"""Launch flags of the AC2 rollout function (``--rollout-function-path examples.ac2.rollout.AC2RolloutFn``).

Sizes the run already names come from the run's own flags: the prefixes per step are ``--rollout-batch-size``,
the continuations per prefix are ``--n-samples-per-prompt``, and the response budget (prefix plus new tokens)
is ``--rollout-max-response-len``.
"""

import argparse

from miles.utils.args.schema import A, Arg, BaseConfig

# The paper's prompt prefill: an empty thinking block, then the answer stem the critic completes.
DEFAULT_CRITIC_PREFILL = "<think>\n\n</think>\n\nQ value: "


class AC2Config(BaseConfig):
    ac2_num_refill: A[
        int | None,
        Arg(
            help=(
                "Fresh problems rolled out to completion each step to refill the replay buffer; they never enter "
                "a loss. Defaults to --rollout-batch-size."
            )
        ),
    ] = None
    ac2_replay_capacity: A[int, Arg(help="FIFO capacity of the replay buffer of finished trajectories.")] = 256
    ac2_chunk_tokens: A[
        int,
        Arg(help="Chunk length b: new tokens per continuation on ready problems, and the grain of prefix cuts."),
    ] = 10000
    ac2_cut_low: A[float, Arg(help="Lowest prefix cut, as a fraction of the trajectory's response tokens.")] = 0.0
    ac2_cut_high: A[float, Arg(help="Highest prefix cut, as a fraction of the trajectory's response tokens.")] = 0.9
    ac2_audit_fraction: A[
        float,
        Arg(help="Fraction of ready prefixes (rounded up) whose continuations still run to completion each step."),
    ] = 0.25
    ac2_pass_threshold: A[
        float,
        Arg(help="Reward at or above which a finished response is correct, and may become a reference solution."),
    ] = 1.0
    ac2_ready_threshold_global: A[
        float,
        Arg(help="A problem can turn ready only while the critic's mean error over the window is below this."),
    ] = 0.20
    ac2_ready_threshold_local: A[
        float,
        Arg(help="A problem turns ready when the critic's error on its prefix this step is below this."),
    ] = 0.18
    ac2_ready_window: A[int, Arg(help="Steps whose critic errors form the global readiness mean.")] = 5
    ac2_ready_require_reference: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help="A problem turns ready only once the reference bank holds a correct solution of it.",
        ),
    ] = True
    ac2_ready_require_nonzero: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help="A problem turns ready only once the critic has predicted a nonzero value on it.",
        ),
    ] = True
    ac2_critic_buffer_capacity: A[int, Arg(help="FIFO capacity of the critic's (prefix, target) records.")] = 1920
    ac2_critic_train_records: A[int, Arg(help="Critic records drawn per step for the aux SFT update.")] = 768
    ac2_critic_min_valid: A[
        int | None,
        Arg(
            help=(
                "Continuations of a prefix that need a valid endpoint value before the prefix yields a critic "
                "target. Defaults to half of --n-samples-per-prompt, rounded up."
            )
        ),
    ] = None
    ac2_critic_train_no_reference: A[
        bool,
        Arg(
            action=argparse.BooleanOptionalAction,
            help=(
                "Train a record that has a reference under both critic prompts at weight 1/2 each; without this, "
                "only its reference prompt at weight 1."
            ),
        ),
    ] = True
    ac2_critic_prefill: A[
        str,
        Arg(help="Text the critic's reply starts with; it completes the value after it."),
    ] = DEFAULT_CRITIC_PREFILL
    ac2_think_close: A[
        str,
        Arg(
            help=(
                "Marker that closes the model's thinking; appended to an attempt that has not closed it before "
                "the critic turn. Empty for a model without a thinking block."
            )
        ),
    ] = "</think>"
    ac2_critic_max_context_len: A[
        int | None,
        Arg(help="Critic prompts longer than this count as invalid predictions and train no row. Unbounded if unset."),
    ] = None
    ac2_seed: A[int, Arg(help="Seed of AC2's own sampling: prefix draws, cuts, audits and critic records.")] = 804001
