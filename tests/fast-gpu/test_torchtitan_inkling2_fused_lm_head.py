from tests.ci.ci_register import register_cuda_ci

# the fused log-prob kernels are Triton and flex attention needs CUDA
register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["torchtitan"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("torchtitan")

from tests.fast.backends.inkling2_tiny import TEXT_CONFIG, as_trained, random_model  # noqa: E402

from miles.backends.training_utils.loss.hub.fused_log_probs import (  # noqa: E402
    fused_lm_head_log_probs_and_entropy,
    fused_log_probs_and_entropy,
)


def _scored_rows(first: int, second: int):
    # the response rows of two packed documents: the last 7 and 6 positions of each
    rows = [*range(first - 7, first - 1), *range(first + second - 6, first + second - 1)]
    return torch.tensor(rows, device="cuda")


def _loss_and_grads(model, log_probs):
    weights = torch.linspace(-1.0, 1.0, log_probs.numel(), device="cuda")
    (log_probs * weights).sum().backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


@pytest.mark.parametrize("tile_rows", [1, 4, 64])
def test_tiled_output_layer_matches_the_full_logits(tmp_path, tile_rows):
    """`--log-probs-backend fused_lm_head` on torchtitan: the model returns hidden states and the loss projects
    tiles of them through `project_logits` (fp32 unembedding, vocab cut, soft cap) under checkpointing. The
    log-probs and every parameter gradient must equal scoring the model's full logits."""
    _, model = random_model(tmp_path)
    model = as_trained(model).cuda().train()
    for param in model.parameters():
        param.requires_grad_(True)
    first, second = 12, 10
    positions = torch.cat([torch.arange(first), torch.arange(second)]).cuda()
    tokens = torch.randint(0, TEXT_CONFIG["unpadded_vocab_size"], (first + second,), device="cuda")
    rows = _scored_rows(first, second)
    targets = tokens[rows + 1]

    model._skip_lm_head = False
    logits = model(tokens, positions)
    full, _ = fused_log_probs_and_entropy(logits, rows, targets, tp_group=None)
    full_grads = _loss_and_grads(model, full)

    model._skip_lm_head = True
    hidden = model(tokens, positions)
    assert hidden.shape[-1] != TEXT_CONFIG["unpadded_vocab_size"]
    tiled, _ = fused_lm_head_log_probs_and_entropy(
        hidden, model.project_logits, rows, targets, tp_group=None, tile_rows=tile_rows
    )
    tiled_grads = _loss_and_grads(model, tiled)

    torch.testing.assert_close(tiled, full)
    assert tiled_grads.keys() == full_grads.keys()
    # the output layer's own weight gradient is summed over the tiles' backwards in its bf16 grad buffer, which
    # rounds each partial sum (0.3-0.5% relative norm error at 1-4 row tiles, exact as one tile); every other
    # gradient flows through the hidden states once and must match exactly
    lm_head_grad = tiled_grads.pop("lm_head.weight").float()
    reference = full_grads.pop("lm_head.weight").float()
    assert (lm_head_grad - reference).norm() <= 1e-2 * reference.norm()
    for name, grad in full_grads.items():
        torch.testing.assert_close(tiled_grads[name], grad, rtol=0, atol=0, msg=name)
