"""Saliency heatmaps for any registered backbone.

Dispatches by `BACKBONE` field in the run's `env.json`:
  * `lvface_vit_b`       — attention rollout (Abnar & Zuidema 2020), with
                            optional gradient weighting (Chefer et al. 2021).
                            Uses an in-place monkey-patch of `_Attention.forward`
                            because the upstream module discards the softmax map.
  * `dinov3`             — re-instantiates the HF model with
                            `attn_implementation="eager"` so `output_attentions=True`
                            returns the full (B, H, N, N) softmax tensors, then
                            runs the same rollout / grad-rollout machinery on them
                            (after stripping the CLS + register-token rows/cols).
  * `inception_resnet_v1`— Grad-CAM (Selvaraju et al. 2017) on `block8`, the
                            last spatial activation before the global pool.
                            Resolution is coarse (3×3 on 160² input) but
                            upsampled bicubically for the overlay.

For ViTs, two visualization modes share the same code path:
  * `rollout`  — class-agnostic, what the model attends to overall.
  * `grad`     — gradient-weighted, attributing either ||emb||² (single image)
                  or cos(probe, reference) when `--reference` is supplied.
  For CNNs, Grad-CAM always uses the gradient target (same ||emb||² / cos
  semantics), so `--method` is ignored.

Usage:
    uv run python scripts/visualize_attention.py \\
        --run-dir runs/trains/2026-05-17T02-11-52_dinov3_cow_faces_v1 \\
        --image path/to/face.png \\
        --output ./heatmaps

    # Probe-vs-reference similarity attribution:
    uv run python scripts/visualize_attention.py \\
        --run-dir runs/trains/.../  --image probe.png --reference gallery.png

    # Or point at a checkpoint directly (run-dir inferred from `.../checkpoints/x.pth`):
    uv run python scripts/visualize_attention.py \\
        --checkpoint runs/trains/.../checkpoints/..._best.pth --image probe.png
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from config.backbone_config import BackboneConfig
from registry import BACKBONES
from tools.lora import apply_lora


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    """Either --run-dir is given, or we derive it from --checkpoint's grandparent
    (`<run_dir>/checkpoints/<file>` → `<run_dir>`)."""
    if args.run_dir is not None:
        return args.run_dir
    if args.checkpoint is None:
        raise SystemExit("provide either --run-dir or --checkpoint")
    ckpt = args.checkpoint
    if ckpt.suffix == ".onnx":
        ckpt = ckpt.with_suffix(".pth")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    # checkpoints/ live one level under the run dir
    return ckpt.parent.parent


def _pick_checkpoint(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        ckpt = explicit
        if ckpt.suffix == ".onnx":
            ckpt = ckpt.with_suffix(".pth")
        return ckpt
    # Prefer the "best" checkpoint
    best = sorted((run_dir / "checkpoints").glob("*_best.pth"))
    if not best:
        raise SystemExit(f"no *_best.pth under {run_dir / 'checkpoints'}")
    return best[0]


def _read_env(run_dir: Path) -> dict[str, str]:
    env_path = run_dir / "env.json"
    if not env_path.exists():
        raise SystemExit(f"missing {env_path} — pass --run-dir of an actual GUI/CLI run")
    with open(env_path) as f:
        return json.load(f)


def _load_backbone(env: dict[str, str], ckpt_path: Path, device: torch.device,
                   force_eager_attn: bool) -> tuple[torch.nn.Module, BackboneConfig, str]:
    """Build the right backbone via the registry, optionally swap DINOv3's HF
    model to eager attention (so `output_attentions=True` works), apply LoRA
    if the checkpoint is wrapped, then load state."""
    backbone_name = env["BACKBONE"]
    # Trigger registry population for backbones.
    importlib.import_module("backbone")

    # Configs were snapshotted into the run dir — paths in env.json are absolute.
    bb_cfg = BackboneConfig(env["BACKBONE_CONFIG"])
    backbone_cls = BACKBONES.get(backbone_name)
    backbone = backbone_cls(bb_cfg).to(device)

    if backbone_name == "dinov3" and force_eager_attn:
        # Swap BEFORE applying LoRA so the adapters wrap the eager Linears.
        from transformers import AutoModel
        backbone.model = AutoModel.from_pretrained(
            backbone.model_name, attn_implementation="eager"
        ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("backbone_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    is_lora_wrapped = any(k.startswith("base_model.model.") for k in sd)

    if is_lora_wrapped:
        with open(env["TRAINER_CONFIG"]) as f:
            tr = yaml.safe_load(f) or {}
        lora = tr.get("lora", {})
        backbone = apply_lora(
            backbone,
            rank=int(lora["rank"]),
            alpha=float(lora["alpha"]),
            target_modules=list(lora["target_modules"]),
            dropout=float(lora.get("dropout", 0.0)),
            modules_to_save=list(lora.get("modules_to_save") or []) or None,
        ).to(device)

    result = backbone.load_state_dict(sd, strict=False)
    if result.missing_keys:
        print(f"[warn] {len(result.missing_keys)} missing keys (first 3): {result.missing_keys[:3]}")
    if result.unexpected_keys:
        print(f"[warn] {len(result.unexpected_keys)} unexpected keys (first 3): {result.unexpected_keys[:3]}")

    backbone.eval()
    return backbone, bb_cfg, backbone_name


def _read_norm(env: dict[str, str]) -> tuple[list[float], list[float]]:
    """Pull val-side mean/std from the run's transformation YAML; fall back to [0.5]*3."""
    tcfg_path = Path(env["TRAIN_VAL_TRANSFORMATION_CONFIG"])
    if not tcfg_path.exists():
        return [0.5] * 3, [0.5] * 3
    with open(tcfg_path) as f:
        tcfg = yaml.safe_load(f) or {}
    norm = (tcfg.get("val") or tcfg.get("train") or {}).get("normalize") or {}
    return list(norm.get("mean", [0.5] * 3)), list(norm.get("std", [0.5] * 3))


def _preprocess(img_path: Path, input_size: tuple[int, int], mean: list[float],
                std: list[float]) -> tuple[torch.Tensor, Image.Image]:
    img = Image.open(img_path).convert("RGB").resize(input_size, Image.BILINEAR)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return tf(img).unsqueeze(0), img


# =============================================================================
# Shared rollout machinery (used by LVFace and DINOv3 paths)
# =============================================================================


def _rollout(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    n = attn_maps[0].shape[-1]
    device = attn_maps[0].device
    result = torch.eye(n, device=device).unsqueeze(0)
    for a in attn_maps:
        a_mean = a.mean(dim=1)  # average over heads → (B, N, N)
        a_aug = a_mean + torch.eye(n, device=device).unsqueeze(0)
        a_aug = a_aug / a_aug.sum(dim=-1, keepdim=True)
        result = a_aug @ result
    return result


def _grad_rollout(attn_maps: list[torch.Tensor], grads: list[torch.Tensor]) -> torch.Tensor:
    n = attn_maps[0].shape[-1]
    device = attn_maps[0].device
    result = torch.eye(n, device=device).unsqueeze(0)
    for a, g in zip(attn_maps, grads, strict=True):
        # ReLU-clip negative contributions so we visualize "what *increases* the target".
        weighted = (a * g).clamp(min=0).mean(dim=1)
        aug = weighted + torch.eye(n, device=device).unsqueeze(0)
        aug = aug / (aug.sum(dim=-1, keepdim=True) + 1e-8)
        result = aug @ result
    return result


def _aggregate_to_grid(rollout: torch.Tensor, num_patches: int) -> np.ndarray:
    """Reduce (1, N, N) → (gh, gh) salience per patch (mean over query rows)."""
    r = rollout[0]
    salience = r.mean(dim=0)
    gh = int(round(num_patches**0.5))
    grid = salience.reshape(gh, gh).detach().cpu().numpy()
    return (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)


# =============================================================================
# Backbone-specific heatmap paths
# =============================================================================


def _lvface_heatmaps(backbone: torch.nn.Module, img_tensor: torch.Tensor, method: str,
                     ref_emb: torch.Tensor | None) -> dict[str, np.ndarray]:
    """Monkey-patch every `_Attention.forward` to stash `attn_map`, run rollout."""
    from backbone.vit import _Attention  # local import — only needed for LVFace

    attn_modules: list[_Attention] = [m for m in backbone.modules() if isinstance(m, _Attention)]

    def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_map = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    for mod in attn_modules:
        mod.forward = patched_forward.__get__(mod, type(mod))

    heatmaps: dict[str, np.ndarray] = {}

    if method in ("rollout", "both"):
        with torch.no_grad():
            _ = backbone(img_tensor)
            attn_maps = [m.attn_map.detach() for m in attn_modules]
        rollout = _rollout(attn_maps)
        heatmaps["rollout"] = _aggregate_to_grid(rollout, attn_maps[0].shape[-1])

    if method in ("grad", "both"):
        backbone.zero_grad(set_to_none=True)
        img_grad = img_tensor.clone().detach().requires_grad_(True)
        emb = backbone(img_grad)
        target, label = _grad_target(emb, ref_emb)
        for m in attn_modules:
            m.attn_map.retain_grad()
        target.backward()
        attn_maps = [m.attn_map.detach() for m in attn_modules]
        grads = [m.attn_map.grad.detach() for m in attn_modules]
        rollout = _grad_rollout(attn_maps, grads)
        heatmaps[label] = _aggregate_to_grid(rollout, attn_maps[0].shape[-1])

    return heatmaps


def _peel_dinov3(backbone: torch.nn.Module):
    """Unwrap LoRA + DinoV3Backbone → return the raw HF DINOv3 model and the
    framework wrapper (for token strategy + feature head)."""
    inner = backbone
    inner = getattr(inner, "base_model", inner)
    inner = getattr(inner, "model", inner)
    # Now `inner` is the DinoV3Backbone instance
    return inner.model, inner


def _dinov3_heatmaps(backbone: torch.nn.Module, img_tensor: torch.Tensor, method: str,
                     ref_emb: torch.Tensor | None) -> dict[str, np.ndarray]:
    """Use HF `output_attentions=True` (requires eager attn); strip CLS+register
    tokens before running rollout so the salience grid is purely patch-level."""
    hf_model, dinov3_wrapper = _peel_dinov3(backbone)
    n_register = int(getattr(hf_model.config, "num_register_tokens", 0))
    n_special = 1 + n_register  # CLS + registers in front of the patch tokens

    def _strip_specials(attns: list[torch.Tensor]) -> list[torch.Tensor]:
        return [a[:, :, n_special:, n_special:] for a in attns]

    heatmaps: dict[str, np.ndarray] = {}

    if method in ("rollout", "both"):
        with torch.no_grad():
            out = hf_model(pixel_values=img_tensor, output_attentions=True)
            attn_maps = _strip_specials(list(out.attentions))
        rollout = _rollout(attn_maps)
        heatmaps["rollout"] = _aggregate_to_grid(rollout, attn_maps[0].shape[-1])

    if method in ("grad", "both"):
        backbone.zero_grad(set_to_none=True)
        img_grad = img_tensor.clone().detach().requires_grad_(True)
        # Manually reproduce DinoV3Backbone.forward so we can keep grad on attentions.
        out = hf_model(pixel_values=img_grad, output_attentions=True)
        hidden = out.last_hidden_state  # (B, 1 + n_register + n_patches, dim)
        if dinov3_wrapper.token_strategy == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = hidden[:, n_special:].mean(dim=1)
        emb = dinov3_wrapper.feature(pooled)

        target, label = _grad_target(emb, ref_emb)
        for a in out.attentions:
            a.retain_grad()
        target.backward()
        attn_maps = _strip_specials([a.detach() for a in out.attentions])
        grads = _strip_specials([a.grad.detach() for a in out.attentions])
        rollout = _grad_rollout(attn_maps, grads)
        heatmaps[label] = _aggregate_to_grid(rollout, attn_maps[0].shape[-1])

    return heatmaps


def _peel_megadescriptor(backbone: torch.nn.Module) -> torch.nn.Module:
    """Unwrap LoRA + MegaDescriptorBackbone → return the raw timm Swin model."""
    inner = backbone
    inner = getattr(inner, "base_model", inner)
    inner = getattr(inner, "model", inner)  # MegaDescriptorBackbone
    return inner.model  # timm Swin model


def _megadescriptor_heatmaps(backbone: torch.nn.Module, img_tensor: torch.Tensor,
                              ref_emb: torch.Tensor | None) -> dict[str, np.ndarray]:
    """Grad-CAM on the last Swin block's output (highest-res semantic features
    just before the global pool). Handles timm Swin's (B, H, W, C) layout
    natively; falls back to (B, L, C) reshape if a future timm version flips
    back to token-sequence outputs.
    """
    timm_model = _peel_megadescriptor(backbone)
    target_layer = timm_model.layers[-1].blocks[-1]

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_m: torch.nn.Module, _i: Any, o: torch.Tensor) -> None:
        activations["v"] = o

    def bwd_hook(_m: torch.nn.Module, _gi: Any, go: tuple[torch.Tensor, ...]) -> None:
        gradients["v"] = go[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)
    try:
        backbone.zero_grad(set_to_none=True)
        img_grad = img_tensor.clone().detach().requires_grad_(True)
        emb = backbone(img_grad)
        target, label = _grad_target(emb, ref_emb)
        target.backward()
    finally:
        fh.remove()
        bh.remove()

    a = activations["v"]
    g = gradients["v"]

    if a.ndim == 4:
        # timm Swin native: (B, H, W, C). GAP over spatial → channel weights.
        weights = g.mean(dim=(1, 2), keepdim=True)            # (B, 1, 1, C)
        cam = (weights * a).sum(dim=-1).clamp(min=0)          # (B, H, W)
    elif a.ndim == 3:
        # Fallback: (B, L, C) with L assumed square.
        b, length, c = a.shape
        h = w = int(round(length ** 0.5))
        if h * w != length:
            raise ValueError(f"non-square Swin token grid: L={length}")
        a4 = a.view(b, h, w, c)
        g4 = g.view(b, h, w, c)
        weights = g4.mean(dim=(1, 2), keepdim=True)
        cam = (weights * a4).sum(dim=-1).clamp(min=0)
    else:
        raise ValueError(f"unexpected activation shape {a.shape} for Swin block")

    cam = cam[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return {f"gradcam (swin last block): {label}": cam}


def _gradcam_heatmaps(backbone: torch.nn.Module, img_tensor: torch.Tensor,
                      ref_emb: torch.Tensor | None,
                      target_layer_name: str = "block8") -> dict[str, np.ndarray]:
    """Standard Grad-CAM on the last spatial activation. CNN-friendly."""
    modules = dict(backbone.named_modules())
    if target_layer_name not in modules:
        raise KeyError(
            f"target layer {target_layer_name!r} not in backbone. Available top-level: "
            f"{list(modules.keys())[:20]}..."
        )
    target_layer = modules[target_layer_name]

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def fwd_hook(_m: torch.nn.Module, _i: Any, o: torch.Tensor) -> None:
        activations["v"] = o

    def bwd_hook(_m: torch.nn.Module, _gi: Any, go: tuple[torch.Tensor, ...]) -> None:
        gradients["v"] = go[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)
    try:
        backbone.zero_grad(set_to_none=True)
        img_grad = img_tensor.clone().detach().requires_grad_(True)
        emb = backbone(img_grad)
        target, label = _grad_target(emb, ref_emb)
        target.backward()
    finally:
        fh.remove()
        bh.remove()

    a = activations["v"]   # (B, C, H, W)
    g = gradients["v"]      # (B, C, H, W)
    weights = g.mean(dim=(2, 3), keepdim=True)  # global-average-pool over spatial dims
    cam = (weights * a).sum(dim=1).clamp(min=0)  # (B, H, W); ReLU keeps positive evidence only
    cam = cam[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return {f"gradcam ({target_layer_name}): {label}": cam}


def _grad_target(emb: torch.Tensor, ref_emb: torch.Tensor | None) -> tuple[torch.Tensor, str]:
    """Pick the scalar to backprop. ||emb||² when no reference; otherwise cos(emb, ref)."""
    if ref_emb is None:
        return (emb * emb).sum(), "grad (||emb||²)"
    sim = F.cosine_similarity(emb, ref_emb, dim=1).sum()
    return sim, f"grad (cos={float(sim.detach().cpu()):.3f})"


# =============================================================================
# Output rendering
# =============================================================================


def _grid_to_overlay(grid: np.ndarray, base_img: Image.Image, alpha: float = 0.45) -> Image.Image:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = base_img.size[1], base_img.size[0]
    grid_t = torch.from_numpy(grid).float().unsqueeze(0).unsqueeze(0)
    grid_up = F.interpolate(grid_t, size=(h, w), mode="bicubic", align_corners=False).squeeze().numpy()
    grid_up = np.clip(grid_up, 0.0, 1.0)
    cmap = plt.get_cmap("jet")
    colored = (cmap(grid_up)[..., :3] * 255).astype(np.uint8)
    return Image.blend(base_img, Image.fromarray(colored), alpha=alpha)


def _save_panel(out_path: Path, base: Image.Image, heatmaps: dict[str, np.ndarray],
                title: str | None = None, simple: bool = False) -> None:
    """Render the figure. `simple=True` drops the raw NxN heatmap columns and
    keeps only [input | overlay_1 | overlay_2 | ...] — much less cluttered when
    you just want to compare attention regions, not pixel-level patch maps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + (len(heatmaps) if simple else 2 * len(heatmaps))
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    if n == 1:
        axes = [axes]
    axes[0].imshow(base)
    axes[0].set_title("input")
    axes[0].axis("off")
    for i, (name, grid) in enumerate(heatmaps.items()):
        overlay = _grid_to_overlay(grid, base)
        if simple:
            ax = axes[1 + i]
            ax.imshow(overlay)
            ax.set_title(name)
            ax.axis("off")
        else:
            axes[1 + 2 * i].imshow(overlay)
            axes[1 + 2 * i].set_title(f"{name} overlay")
            axes[1 + 2 * i].axis("off")
            axes[2 + 2 * i].imshow(grid, cmap="jet")
            gh, gw = grid.shape
            axes[2 + 2 * i].set_title(f"{name} ({gh}x{gw})")
            axes[2 + 2 * i].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="run directory containing env.json + configs/ + checkpoints/")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="explicit .pth path (run-dir inferred from grandparent). "
                        "If you pass an .onnx, the .pth sibling is used.")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--reference", type=Path, default=None,
                   help="optional second image — attributes cos(probe, ref) instead of ||emb||²")
    p.add_argument("--method", choices=["rollout", "grad", "both"], default="both",
                   help="ignored for CNN backbones (always Grad-CAM)")
    p.add_argument("--cnn-target-layer", default="repeat_2",
                   help="CNN Grad-CAM target layer name. Default `repeat_2` (8×8 on "
                        "InceptionResNetV1 @ 160²) — strictly canonical Grad-CAM would "
                        "use the last conv block (`block8`, but it's only 3×3). "
                        "`repeat_2` gives much higher-resolution maps at minor semantic loss.")
    p.add_argument("--output", type=Path, default=Path("./heatmaps"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--title", default=None, help="optional title for the saved panel")
    p.add_argument("--simple", action="store_true",
                   help="drop raw NxN grid columns; keep only [input | overlay_1 | overlay_2 ...]")
    args = p.parse_args()

    run_dir = _resolve_run_dir(args)
    env = _read_env(run_dir)
    ckpt_path = _pick_checkpoint(run_dir, args.checkpoint)
    backbone_name = env["BACKBONE"]
    print(f"[info] run_dir={run_dir.name}  backbone={backbone_name}  ckpt={ckpt_path.name}")

    device = torch.device(args.device)
    force_eager = backbone_name == "dinov3"
    backbone, bb_cfg, _ = _load_backbone(env, ckpt_path, device, force_eager_attn=force_eager)

    inp_size = tuple(int(s) for s in bb_cfg.input_size)
    mean, std = _read_norm(env)
    img_tensor, img_pil = _preprocess(args.image, inp_size, mean, std)
    img_tensor = img_tensor.to(device)

    ref_emb = None
    if args.reference is not None:
        ref_tensor, _ = _preprocess(args.reference, inp_size, mean, std)
        with torch.no_grad():
            ref_emb = backbone(ref_tensor.to(device))

    if backbone_name == "lvface_vit_b":
        heatmaps = _lvface_heatmaps(backbone, img_tensor, args.method, ref_emb)
    elif backbone_name == "dinov3":
        heatmaps = _dinov3_heatmaps(backbone, img_tensor, args.method, ref_emb)
    elif backbone_name == "inception_resnet_v1":
        heatmaps = _gradcam_heatmaps(backbone, img_tensor, ref_emb,
                                     target_layer_name=args.cnn_target_layer)
    elif backbone_name == "megadescriptor":
        heatmaps = _megadescriptor_heatmaps(backbone, img_tensor, ref_emb)
    else:
        raise NotImplementedError(f"no heatmap path for backbone {backbone_name!r}")

    out_file = args.output / f"{args.image.stem}__{run_dir.name}.png"
    _save_panel(out_file, img_pil, heatmaps, title=args.title or f"{run_dir.name}  ({backbone_name})",
                simple=args.simple)
    print(f"[ok] wrote {out_file}")


if __name__ == "__main__":
    main()
