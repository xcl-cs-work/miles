import numpy as np
import pytest
from examples.ac2.critic import (
    CriticPrompt,
    critic_instruction,
    extract_reference,
    format_grid_value,
    parse_critic_value,
    round_to_grid,
)


class CharTokenizer:
    """One token per character, and a chat template whose turns read ``[role]content[end]``."""

    eos_token_id = 1

    def __init__(self, reply_start: str = "[assistant]"):
        self.reply_start = reply_start

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(chr(i) for i in ids if i != self.eos_token_id)

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False) -> str:
        text = "".join(f"[{m['role']}]{m['content']}[end]" for m in messages)
        return text + (self.reply_start if add_generation_prompt else "")


class TestGrid:
    @pytest.mark.parametrize(("value", "grid"), [(0.04, 0.0), (0.26, 0.3), (0.25, 0.2), (0.35, 0.3), (0.96, 1.0)])
    def test_rounds_to_the_nearest_tenth_and_ties_go_down(self, value, grid):
        assert round_to_grid(value) == grid

    @pytest.mark.parametrize(("value", "text"), [(0.0, "0"), (0.3, "0.3"), (1.0, "1")])
    def test_formats_the_trained_answer(self, value, text):
        assert format_grid_value(value) == text

    @pytest.mark.parametrize(
        ("reply", "value"),
        [("0.3", 0.3), (" 1", 1.0), ("1.0\n", 1.0), (".5", 0.5), ("0", 0.0), ("0.7 because", 0.7)],
    )
    def test_parses_a_leading_grid_value(self, reply, value):
        assert parse_critic_value(reply) == value

    @pytest.mark.parametrize("reply", ["0.75", "Q value", "", "2", "-0.1"])
    def test_an_off_grid_or_missing_value_is_invalid_not_coerced(self, reply):
        assert parse_critic_value(reply) is None


class TestCriticPrompt:
    def _prompt(self, **kwargs) -> CriticPrompt:
        return CriticPrompt(CharTokenizer(**kwargs), think_close="</think>", prefill="<think>\n\n</think>\n\nQ: ")

    def test_an_open_attempt_gets_its_thinking_closed_before_the_critic_turn(self):
        tokenizer = CharTokenizer()
        context = self._prompt().context_ids(tokenizer.encode("P"), tokenizer.encode("<think>\nhmm"), None)
        assert tokenizer.decode(context) == (
            "P<think>\nhmm</think>[end][user]"
            + critic_instruction(None)
            + "[end][assistant]<think>\n\n</think>\n\nQ: "
        )

    def test_a_closed_attempt_is_kept_as_is(self):
        tokenizer = CharTokenizer()
        attempt = tokenizer.encode("<think>a</think>so 5")
        context = self._prompt().context_ids(tokenizer.encode("P"), np.asarray(attempt, dtype=np.int32), "5")
        text = tokenizer.decode(context)
        assert text.startswith("P<think>a</think>so 5[end][user]") and text.count("</think>") == 2
        assert "reference correct solution here:\n\n5\n\n" in text
        assert all(type(token) is int for token in context)

    def test_a_template_that_opens_thinking_does_not_open_it_twice(self):
        tokenizer = CharTokenizer(reply_start="[assistant]<think>\n")
        context = self._prompt(reply_start="[assistant]<think>\n").context_ids([], [], None)
        assert tokenizer.decode(context).endswith("[assistant]<think>\n\n</think>\n\nQ: ")

    def test_the_target_is_the_value_then_the_end_of_turn(self):
        assert self._prompt().target_ids(0.3) == [ord("0"), ord("."), ord("3"), CharTokenizer.eos_token_id]


@pytest.mark.parametrize(
    ("response", "reference"),
    [("<think>work</think>\n\nThe answer is 5. ", "The answer is 5."), ("no thinking block", "no thinking block")],
)
def test_the_reference_is_the_text_after_the_last_thinking_close(response, reference):
    assert extract_reference(response, think_close="</think>") == reference
