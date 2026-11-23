"""AC2's generative critic: the policy model itself, asked in a new user turn how much credit an attempt will earn.

The critic's input is the problem, the partial attempt (its thinking closed if still open) and a user turn with
the instruction below, optionally carrying a reference solution; the reply is prefilled up to ``Q value: `` and the
model completes a value on the grid {0, 0.1, ..., 1}.
"""

import re
from collections.abc import Sequence

import numpy as np
from transformers import PreTrainedTokenizerBase

GRID = tuple(i / 10 for i in range(11))

_INSTRUCTION_HEAD = (
    "Pause here and estimate how much credit this attempt will earn if you continue it to completion within its "
    "remaining token budget: 0 means no credit, 1 means full credit for a complete and correct solution, and values "
    "in between mean partial credit."
)
_REFERENCE_CLAUSE = " You should refer to a reference correct solution here:\n\n{reference}"
_INSTRUCTION_TAIL = (
    "\n\nDon't think; answer immediately with a single line and nothing else:\n\nQ value: z\n\n"
    "where z is one of 0, 0.1, ..., 1. The first characters of your response must be `Q value:`."
)

_LEADING_NUMBER = re.compile(r"^\s*(1(?:\.0+)?|0(?:\.\d+)?|\.\d+)")

# Distinct markers to cut the chat template's turn boundaries out of a rendered conversation.
_USER_MARK, _ASSISTANT_MARK, _CRITIC_MARK = "AC2USERMARK", "AC2ASSISTANTMARK", "AC2CRITICMARK"


def round_to_grid(value: float) -> float:
    """The nearest grid value; a tie goes to the lower one."""
    return min(GRID, key=lambda g: (abs(g - value), g))


def format_grid_value(value: float) -> str:
    """``0``, ``0.1``, ..., ``1``: the form the critic is trained to emit."""
    return f"{value:.1f}".rstrip("0").rstrip(".")


def parse_critic_value(text: str) -> float | None:
    """The value the critic's reply starts with, or None when it is missing or off the grid (never coerced)."""
    match = _LEADING_NUMBER.match(text)
    if match is None:
        return None
    value = float(match.group(1))
    grid_value = round_to_grid(value)
    return grid_value if abs(grid_value - value) <= 1e-6 else None


def critic_instruction(reference: str | None) -> str:
    reference_clause = "" if reference is None else _REFERENCE_CLAUSE.format(reference=reference)
    return _INSTRUCTION_HEAD + reference_clause + _INSTRUCTION_TAIL


def extract_reference(response: str, *, think_close: str) -> str:
    """The final solution of a response: its text after the last thinking close, or all of it without one."""
    if think_close and think_close in response:
        response = response.rsplit(think_close, 1)[1]
    return response.strip()


class CriticPrompt:
    """Builds the critic's token sequences for one tokenizer: query contexts and trained targets."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *, think_close: str, prefill: str):
        if tokenizer.eos_token_id is None:
            raise ValueError("the critic target ends with the tokenizer's eos token, which this tokenizer lacks")
        self._tokenizer = tokenizer
        between_turns, reply_start = _turn_boundaries(tokenizer)
        if reply_start.endswith("<think>\n") and prefill.startswith("<think>\n"):
            # the template already opened the thinking block the prefill opens
            prefill = prefill[len("<think>\n") :]
        self._think_close_ids = self._encode(think_close) if think_close else []
        self._between_turns_ids = self._encode(between_turns)
        self._reply_ids = self._encode(reply_start + prefill)
        self._eos_id = tokenizer.eos_token_id

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def context_ids(self, prompt_ids: Sequence[int], attempt_ids: Sequence[int], reference: str | None) -> list[int]:
        """The critic's input: problem, attempt (thinking closed), critic turn, and the prefilled reply."""
        prompt_ids, attempt_ids = _int_list(prompt_ids), _int_list(attempt_ids)
        close = [] if _contains(attempt_ids, self._think_close_ids) else self._think_close_ids
        return (
            prompt_ids
            + attempt_ids
            + close
            + self._between_turns_ids
            + self._encode(critic_instruction(reference))
            + self._reply_ids
        )

    def target_ids(self, value: float) -> list[int]:
        """The supervised continuation of a critic context: the grid value and the end of the turn."""
        return self._encode(format_grid_value(value)) + [self._eos_id]


def _turn_boundaries(tokenizer: PreTrainedTokenizerBase) -> tuple[str, str]:
    """The chat template's text between an assistant reply and the next user content, and after that content."""
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": _USER_MARK},
            {"role": "assistant", "content": _ASSISTANT_MARK},
            {"role": "user", "content": _CRITIC_MARK},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    after_assistant = rendered.split(_ASSISTANT_MARK, 1)[1]
    between_turns, reply_start = after_assistant.split(_CRITIC_MARK, 1)
    return between_turns, reply_start


def _int_list(ids: Sequence[int]) -> list[int]:
    """Plain ints, as request payloads and row tokens need, from a list or an int32 array."""
    return np.asarray(ids, dtype=np.int64).tolist()


def _contains(sequence: list[int], pattern: list[int]) -> bool:
    if not pattern:
        return True
    n = len(pattern)
    first = pattern[0]
    return any(sequence[i] == first and list(sequence[i : i + n]) == pattern for i in range(len(sequence) - n + 1))
