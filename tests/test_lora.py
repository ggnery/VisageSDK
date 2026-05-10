"""Unit tests for `src/tools/lora.py`.

These exercise `apply_lora` / `lora_trainable_summary` in isolation against
tiny synthetic modules so the coverage doesn't depend on the full
InceptionResNetV1 backbone (which is exercised end-to-end in
`test_trainer.py::TestLoRAIntegration`).

All assertions target framework invariants we rely on at the call site:
- PEFT freezes everything except LoRA params.
- LoRA is initialized so the wrapped model's forward equals the base
  model's at step 0 (B = 0). Without that property, plugging LoRA in mid-
  training would silently shift predictions.
- Param counts scale linearly with rank — useful for budget arithmetic.
- Adapter weights round-trip through save/load.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft.peft_model import PeftModel

from tools.lora import apply_lora, lora_trainable_summary


class _TwoLinearModel(nn.Module):
    """Bare-minimum target for LoRA tests: two named Linear layers + a
    BatchNorm we deliberately don't target. Mirrors the trainer's actual
    targeting pattern (specific named modules, not "all linears")."""

    def __init__(self, in_dim: int = 8, hidden: int = 12, out_dim: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden, bias=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.head = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.bn(torch.relu(self.proj(x))))


def _fresh_model() -> _TwoLinearModel:
    torch.manual_seed(0)
    m = _TwoLinearModel()
    m.eval()  # so the BN running stats stay fixed across tests
    return m


# =============================================================================
# Wrapping behavior
# =============================================================================


class TestApplyLoraWraps:
    def test_returns_peft_model(self):
        m = _fresh_model()
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        assert isinstance(wrapped, PeftModel)

    def test_attribute_access_proxies_to_base(self):
        """The Trainer reads `self.backbone.input_size`, `embedding_size`,
        etc. PEFT's __getattr__ forwards unknown attrs to the base model;
        breaking that contract would break the trainer pipeline."""
        m = _fresh_model()
        m.input_size = [160, 160]  # type: ignore[attr-defined]
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        assert wrapped.input_size == [160, 160]  # type: ignore[attr-defined]


# =============================================================================
# Freeze / trainable structure
# =============================================================================


class TestApplyLoraFreezeStructure:
    def test_only_lora_params_trainable(self):
        m = _fresh_model()
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        trainable_names = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        assert trainable_names, "expected at least one trainable LoRA tensor"
        assert all("lora_" in n for n in trainable_names), (
            f"non-LoRA params trainable: {trainable_names}"
        )

    def test_targeted_modules_get_lora_layers(self):
        """When two modules are targeted, both get their own A/B pair."""
        m = _fresh_model()
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["proj", "head"])
        trainable_names = [n for n, p in wrapped.named_parameters() if p.requires_grad]
        # PEFT names them with the targeted module path embedded.
        assert any("proj" in n for n in trainable_names)
        assert any("head" in n for n in trainable_names)

    def test_non_targeted_modules_are_frozen(self):
        """`bn` is intentionally NOT in target_modules; its params must
        stay frozen so the BN running stats / affine params don't drift."""
        m = _fresh_model()
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        bn_params = [p for n, p in wrapped.named_parameters() if "bn" in n]
        assert bn_params, "BN params disappeared from the wrapped model"
        assert all(not p.requires_grad for p in bn_params)


# =============================================================================
# Numerical invariants
# =============================================================================


class TestApplyLoraNumerics:
    def test_initial_forward_equals_base(self):
        """LoRA's standard init (A=kaiming, B=0) means at step 0 the
        wrapped model's forward must equal the base model's. If this
        regresses, every LoRA-enabled run silently starts from a
        different point than the loaded checkpoint."""
        m = _fresh_model()
        x = torch.randn(2, 8)
        with torch.no_grad():
            base_out = m(x)
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        wrapped.eval()
        with torch.no_grad():
            lora_out = wrapped(x)
        assert torch.allclose(base_out, lora_out, atol=1e-6)

    def test_b_matrix_initialized_to_zero(self):
        """Document the LoRA-init invariant explicitly so a peft upgrade
        that ever changes it is caught immediately rather than as a
        mysterious training divergence."""
        m = _fresh_model()
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        b_tensors = [p for n, p in wrapped.named_parameters() if ".lora_B." in n]
        assert b_tensors, "expected at least one lora_B tensor"
        for p in b_tensors:
            assert torch.all(p == 0), "lora_B must be zero-initialized"

    def test_a_matrix_not_zero(self):
        """Sanity-check that lora_A is non-trivially initialized — pairing
        zero A with zero B would freeze the LoRA path entirely."""
        m = _fresh_model()
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        a_tensors = [p for n, p in wrapped.named_parameters() if ".lora_A." in n]
        assert a_tensors
        assert any(torch.any(p != 0) for p in a_tensors)


# =============================================================================
# Param-count budget
# =============================================================================


class TestApplyLoraParamCounts:
    def test_param_count_scales_with_rank(self):
        """LoRA on Linear(in, out) adds (in + out) * rank params per
        targeted module. Doubling rank should ~double the trainable count."""
        wrapped_r4 = apply_lora(_fresh_model(), rank=4, alpha=8.0, target_modules=["head"])
        wrapped_r8 = apply_lora(_fresh_model(), rank=8, alpha=8.0, target_modules=["head"])
        n4 = sum(p.numel() for p in wrapped_r4.parameters() if p.requires_grad)
        n8 = sum(p.numel() for p in wrapped_r8.parameters() if p.requires_grad)
        assert n8 == 2 * n4

    def test_exact_count_for_known_shapes(self):
        """head is Linear(12 -> 4). With rank=4, lora_A is (4, 12) and
        lora_B is (4, 4) → 48 + 16 = 64 trainable params. Pinning the
        exact number guards against PEFT silently changing decomposition."""
        wrapped = apply_lora(_fresh_model(), rank=4, alpha=8.0, target_modules=["head"])
        n_train = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        assert n_train == (12 * 4 + 4 * 4)


# =============================================================================
# lora_trainable_summary helper
# =============================================================================


class TestLoraTrainableSummary:
    def test_unwrapped_model_counts_all_as_trainable(self):
        m = _fresh_model()
        trainable, total = lora_trainable_summary(m)
        assert trainable == total
        assert total == sum(p.numel() for p in m.parameters())

    def test_wrapped_model_reports_only_lora_trainable(self):
        wrapped = apply_lora(_fresh_model(), rank=4, alpha=8.0, target_modules=["head"])
        trainable, total = lora_trainable_summary(wrapped)
        assert trainable < total
        # Cross-check against the manual computation.
        manual = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        assert trainable == manual


# =============================================================================
# modules_to_save (full fine-tune) — escape hatch for the embedding head
# =============================================================================


class TestApplyLoraModulesToSave:
    def test_listed_module_becomes_fully_trainable(self):
        """`modules_to_save=["proj"]` must mark every parameter under that
        module as trainable, on top of the LoRA-only training of `head`.
        Without that, the user-facing override is silently a no-op."""
        m = _fresh_model()
        wrapped = apply_lora(
            m, rank=4, alpha=8.0,
            target_modules=["head"],
            modules_to_save=["proj"],
        )
        # Filter to params under `proj` (excluding LoRA-style names).
        proj_trainable = [
            (n, p)
            for n, p in wrapped.named_parameters()
            if "proj" in n and "lora_" not in n and p.requires_grad
        ]
        assert proj_trainable, (
            "modules_to_save did not unfreeze `proj` — PEFT silently "
            "ignored the request, so the feature head stays frozen "
            "(the bug we're guarding against)."
        )

    def test_default_keeps_non_targeted_modules_frozen(self):
        """Without modules_to_save, only LoRA params are trainable —
        sanity check that the feature for this fix is genuinely opt-in
        and we haven't regressed the default behavior."""
        wrapped = apply_lora(
            _fresh_model(), rank=4, alpha=8.0, target_modules=["head"],
        )
        proj_trainable = [
            n for n, p in wrapped.named_parameters()
            if "proj" in n and "lora_" not in n and p.requires_grad
        ]
        assert proj_trainable == []

    def test_modules_to_save_increases_param_count(self):
        """Adding modules_to_save increases the trainable count by the
        full size of the listed modules (not just rank * (in+out))."""
        baseline = apply_lora(
            _fresh_model(), rank=4, alpha=8.0, target_modules=["head"],
        )
        with_save = apply_lora(
            _fresh_model(), rank=4, alpha=8.0,
            target_modules=["head"], modules_to_save=["proj"],
        )
        n_baseline = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
        n_with_save = sum(p.numel() for p in with_save.parameters() if p.requires_grad)
        # `proj` is Linear(8 -> 12) bias=False → 96 weight params unfrozen.
        # Plus PEFT typically duplicates `modules_to_save` modules into a
        # separate `modules_to_save.default` copy (so the wrapped delta is
        # comparable to baseline + ~2 * proj_params).
        assert n_with_save > n_baseline + 90


# =============================================================================
# Save / load roundtrip
# =============================================================================


class TestApplyLoraStateRoundtrip:
    def test_state_dict_roundtrip_preserves_outputs(self, tmp_path):
        """A saved + reloaded LoRA-wrapped model must reproduce identical
        outputs. Catches both PEFT save_pretrained issues and any silent
        re-initialization on load."""
        m = _fresh_model()
        wrapped = apply_lora(m, rank=4, alpha=8.0, target_modules=["head"])
        wrapped.eval()

        # Bump lora_B so the LoRA path actually contributes; otherwise the
        # roundtrip equality is trivial (forward equals the base in both
        # directions regardless of save/load correctness).
        with torch.no_grad():
            for n, p in wrapped.named_parameters():
                if ".lora_B." in n:
                    p.fill_(0.05)

        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = wrapped(x).clone()

        # Round-trip through state_dict (full state, not adapter-only —
        # matches what the framework's checkpoint format uses).
        state = wrapped.state_dict()

        m2 = _fresh_model()
        wrapped2 = apply_lora(m2, rank=4, alpha=8.0, target_modules=["head"])
        wrapped2.load_state_dict(state)
        wrapped2.eval()

        with torch.no_grad():
            actual = wrapped2(x)
        assert torch.allclose(expected, actual, atol=1e-6)
