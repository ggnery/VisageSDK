# VisageSDK

A modular face recognition training framework. Pick a backbone, a loss, a
dataset, and either drive training from a single `.env` file or from the
included Streamlit GUI. The framework ships with the metrics that actually
matter for face recognition (LFW 10-fold accuracy, TAR@FAR, ROC-AUC, EER,
mAP, rank-N, CMC) and a fine-tuning workflow that supports layer freezing
and discriminative learning rates.

## Table of contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Streamlit GUI](#streamlit-gui)
- [Project layout](#project-layout)
- [Configuration system](#configuration-system)
- [Available components](#available-components)
- [Trainer YAML reference](#trainer-yaml-reference)
- [Fine-tuning (freeze + discriminative LRs)](#fine-tuning-freeze--discriminative-lrs)
- [Evaluation](#evaluation)
- [TensorBoard](#tensorboard)
- [Adding a custom component](#adding-a-custom-component)
- [Development](#development)

## Highlights

- **Registry-driven**: pick a backbone or loss by name (`triplet`, `mobilenetv3`, ...) instead of editing import paths.
- **YAML configs without boilerplate**: every YAML key is exposed as an attribute via `BaseConfig.__getattr__`, so adding a parameter is a one-line YAML change — no parallel `Config` subclass required.
- **Fine-tuning out of the box**: freeze backbone layers, attach discriminative learning rates, and progressively unfreeze on a schedule.
- **Real evaluation**: `eval.py` runs an evaluator against any checkpoint and reports LFW 10-fold accuracy, TAR@FAR (1e-3 / 1e-4 / 1e-5 / 1e-6), ROC-AUC, EER, mAP, rank-N, and CMC.
- **Modern training loop**: AMP (`float16` / `bfloat16`), gradient clipping, deterministic seeding, TensorBoard scalars, and inline periodic evaluation.
- **GUI**: Streamlit app to configure, launch, monitor, and evaluate runs from the browser.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/). Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the environment:

```bash
git clone <repository-url>
cd VisageSDK
uv sync
```

`uv sync` creates a `.venv/` and installs the exact versions pinned in `uv.lock`. Python 3.12+ is required (see `pyproject.toml`).

To add or upgrade dependencies:

```bash
uv add <package>          # add a new dependency
uv lock --upgrade         # refresh the lock file
```

## Quick start

```bash
cp .env.example .env
# edit .env to point at your dataset / pick backbone / loss
uv run python train.py
```

`train.py` reads the env vars, instantiates everything via the registry, and starts training. Checkpoints, the resolved config snapshot, and TensorBoard events go to `<checkpoint_save_dir>/`.

A minimal `.env` looks like:

```env
BACKBONE=mobilenetv3
BACKBONE_CONFIG=./configs/backbone/mobilenetv3.yaml

TRAIN_VAL_DATASET=image_folder
TRAIN_VAL_DATASET_CONFIG=./configs/dataset/train_val/casia_webface.yaml

TRAIN_TRANSFORMATION=casia_webface_train
VAL_TRANSFORMATION=casia_webface_val
TRAIN_VAL_TRANSFORMATION_CONFIG=./configs/transformation/train_val/casia_webface.yaml

LOSS=margin_cosine
LOSS_CONFIG=./configs/loss/margin_cosine.yaml

TRAINER_CONFIG=./configs/trainer/mobilenetv3_from_scratch.yaml

# Optional
EARLY_STOPPER=adaptive
EARLY_STOPPER_CONFIG=./configs/early_stopper/adaptive.yaml
# SAMPLER=facenet
# SAMPLER_CONFIG=./configs/batch_sampler/facenet.yaml
```

See `.env.example` for a documented reference and `.env.all` for snippets of every registered variant.

## Streamlit GUI

The GUI lets you pick components, edit their YAMLs inline, launch training, watch live charts, and run evaluations — all in the browser.

```bash
uv run streamlit run gui/app.py
```

It exposes four tabs:

- **Configure & Train** — registry-backed dropdowns for every component, inline YAML editors, quick overrides for `num_epochs` / `device` / `seed` / AMP / TensorBoard, and an optional `periodic_eval` block. Clicking *Launch Training* snapshots the YAMLs into `runs/<timestamp>/configs/` and spawns `train.py`.
- **Monitor Train** — picks a training run and reads the trainer's TensorBoard events live: loss/lr/eval/train_stats/val_stats line charts plus the tail of `train.log`. Auto-refresh and a Stop button are built in.
- **Monitor Eval** — picks an eval run and renders the JSON metric bundle (headline cards, ROC curve, score distributions) plus the tail of `eval.log`.
- **Evaluate** — picks a checkpoint and runs `eval.py` against any registered evaluator, displaying the JSON results.

## Project layout

```
VisageSDK/
├── pyproject.toml                  # uv project file
├── uv.lock                         # pinned dependency tree
├── train.py                        # training entry point
├── eval.py                         # evaluation entry point
├── .env.example                    # train.py env vars
├── .env.eval.example               # eval.py env vars
├── configs/                        # YAMLs grouped by component
│   ├── backbone/                   # input_size, embedding_size, model knobs
│   ├── loss/
│   ├── dataset/{train_val,eval}/
│   ├── transformation/{train_val,eval}/
│   ├── batch_sampler/
│   ├── early_stopper/
│   ├── evaluator/
│   └── trainer/                    # optimizer, schedule, AMP, freeze, periodic_eval, ...
├── src/
│   ├── registry.py                 # BACKBONES / LOSSES / DATASETS / ... registries
│   ├── config/                     # one Base*Config per category, no leaf classes
│   ├── backbone/                   # InceptionResNetV1/V2, InceptionV4, MobileNetV3
│   ├── loss/                       # Triplet, Center, CrossEntropy, MarginCosineProduct
│   ├── dataset/{train_val,eval}/   # ImageFolder / LFW pairs / Identification
│   ├── transformation/{train_val,eval}/
│   ├── batch_sampler/              # FacenetBatchSampler
│   ├── early_stopper/              # AdaptiveEarlyStopper
│   ├── evaluator/                  # Verification, Identification
│   ├── tools/                      # optimizer, scheduler, freezer, metrics, seed, builders
│   └── trainer/                    # Trainer with AMP / grad clip / TB / periodic eval
├── gui/
│   ├── app.py                      # Streamlit app (3 tabs)
│   └── run_manager.py              # subprocess launcher + TB event reader
└── runs/                           # GUI launches drop snapshots here (gitignored)
```

## Configuration system

There are exactly two layers:

1. **Component selection** — env vars (or the GUI) pick which class to instantiate from the registry: `BACKBONE=mobilenetv3`, `LOSS=triplet`, etc.
2. **Component parameters** — a YAML per component, pointed at by `*_CONFIG` env vars.

Every YAML key is automatically exposed as an attribute on its config object. For example, `configs/loss/triplet.yaml`:

```yaml
device: cuda
margin: 0.2
```

is consumed inside `TripletLoss.__init__` simply as `self.margin = loss_config.margin`. There is no per-component `Config` subclass to write — `BaseLossConfig` (and its peers) handle the dispatch via `__getattr__`. Cross-component fields (`embedding_size`, `num_classes`, `input_size`) are injected by the builder.

The trainer uses a slightly richer config (`TrainerConfig`) because it has structured nested blocks. See [Trainer YAML reference](#trainer-yaml-reference).

## Available components

| Kind | Registered names |
| --- | --- |
| Backbones | `inception_resnet_v1`, `inception_resnet_v2`, `inception_v4`, `mobilenetv3`, `lvface_vit_b`, `dinov3`, `megadescriptor` |
| Losses | `triplet`, `center`, `cross_entropy`, `margin_cosine`, `arcface` |
| Train/val datasets | `image_folder` (single class with `split="train"`/`"val"`) |
| Eval datasets | `lfw_pairs`, `identification` |
| Transformations | `vgg_face2_train`, `vgg_face2_val`, `casia_webface_train`, `casia_webface_val`, `lfw_eval` |
| Samplers | `facenet` |
| Early stoppers | `adaptive` |
| Evaluators | `verification`, `identification` |

## Trainer YAML reference

```yaml
optimizer:
  type: SGD                # SGD | Adam | AdamW | RMSprop
  params:
    lr: 0.1
    momentum: 0.9
    weight_decay: 5.0e-4
  param_groups:            # OPTIONAL: discriminative LRs
    - pattern: "backbone.features.[0-2].*"
      lr: 1.0e-5
    - pattern: "backbone.features.*"
      lr: 1.0e-4
    - pattern: "loss.*"
      lr: 1.0e-3

lr_schedule:
  type: StepLR             # StepLR | MultiStepLR | StairLR | ReduceLROnPlateau
  params: { step_size: 10, gamma: 0.5 }

dataloader:
  train: { batch_size: 128, shuffle: true,  num_workers: 8 }
  val:   { batch_size: 128, shuffle: false, num_workers: 8 }

num_epochs: 30
device: cuda

checkpoint:
  save: { dir: ./checkpoints/run, frequency: 5 }
  load:
    path: null              # set to a .pth to resume / fine-tune
    backbone: true
    loss: true
    scheduler: true
    optimizer: true

# --- All blocks below are optional ---

freeze:
  except: ["last_linear*", "last_bn*"]   # OR `patterns: [...]` to freeze a list explicitly
  unfreeze_at_epoch:
    3: ["features.[8-9]*"]
    6: ["features.[3-7]*"]

seed: 42
deterministic: false        # forces cuDNN deterministic mode

amp:
  enabled: true             # auto-disabled on CPU
  dtype: float16            # float16 | bfloat16

gradient_clip:
  max_norm: 5.0             # null disables
  norm_type: 2.0

logging:
  tensorboard: true
  log_dir: null             # default: <checkpoint.save.dir>/runs/<timestamp>

periodic_eval:
  enabled: true
  every_n_epochs: 5
  dataset: lfw_pairs
  dataset_config: ./configs/dataset/eval/lfw_pairs.yaml
  transformation: lfw_eval
  transformation_config: ./configs/transformation/eval/lfw.yaml
  evaluator: verification
  evaluator_config: ./configs/evaluator/lfw_verification.yaml
```

`configs/trainer/finetune_example.yaml` is a self-documenting template that exercises every block.

## Fine-tuning (freeze + discriminative LRs)

Patterns are `fnmatch` globs, but freeze and optimizer rules match **different name spaces** — don't mix them up:

- **`freeze.patterns` / `freeze.except` / `freeze.unfreeze_at_epoch`** match the backbone's **bare** `named_parameters()` keys — **no prefix** (e.g. `last_linear*`, `last_bn*`, `block8.*`). A `freeze.except` whose patterns match nothing freezes the *entire* backbone (head included), so training stalls on top of frozen features — the freezer logs a warning when this happens.
- **`optimizer.param_groups`** patterns match **prefixed** names: `backbone.*` for backbone params, `loss.*` for loss params (e.g. `backbone.features.*`, `loss.*`).

In short: `freeze.except: ["last_linear*"]` (unprefixed) vs `param_groups: [{pattern: "backbone.features.*", ...}]` (prefixed).

- `freeze.patterns`: list of patterns to freeze.
- `freeze.except`: freeze every parameter that does *not* match any of these patterns (handy for "freeze the backbone except the head").
- `freeze.unfreeze_at_epoch`: `{epoch: [patterns]}` — patterns released at the start of that epoch.
- `optimizer.param_groups`: list of `{pattern, lr, ...}` overrides assigned in order; unmatched parameters fall into a default group at `optimizer.params.lr`.

Frozen parameters stay in the optimizer (with `requires_grad=False`), so unfreezing them mid-run picks up the existing optimizer state cleanly. The trainer logs `trainable / total` parameters whenever the freeze state changes.

## Evaluation

`eval.py` is the standalone evaluation entry point. Configure it through `.env.eval.example`:

```env
BACKBONE=inception_resnet_v1
BACKBONE_CONFIG=./configs/backbone/inception_resnet_v1.yaml
CHECKPOINT_PATH=./checkpoints/.../best.pth

EVAL_DATASET=lfw_pairs                  # or identification
EVAL_DATASET_CONFIG=./configs/dataset/eval/lfw_pairs.yaml

EVAL_TRANSFORMATION=lfw_eval
EVAL_TRANSFORMATION_CONFIG=./configs/transformation/eval/lfw.yaml

EVALUATOR=verification                  # or identification
EVALUATOR_CONFIG=./configs/evaluator/lfw_verification.yaml
```

```bash
uv run python eval.py
```

A JSON file with the metric bundle is dropped next to the checkpoint.

### Verification metrics (`evaluator: verification`, dataset: `lfw_pairs`)

Reads the standard LFW `pairs.txt` format (`<n_folds> <n_pairs_per_fold>` header, then alternating same / different pair blocks). Each unique image is encoded once.

- `lfw_accuracy_mean` / `lfw_accuracy_std` — official 10-fold protocol (per-fold threshold tuned on the rest)
- `lfw_threshold_mean` — average threshold picked across folds
- `best_threshold_global` / `best_accuracy_global` — single threshold over the whole set
- `roc_auc`, `eer` (+ threshold)
- `tar@far=1e-03`, `tar@far=1e-04`, `tar@far=1e-05`, `tar@far=1e-06` (`far_targets` configurable per evaluator YAML)

### Identification metrics (`evaluator: identification`, dataset: `identification`)

Expected layout:

```
<eval_dir>/
├── gallery/<person>/<images>
└── probe/<person>/<images>
```

Reports:

- `rank_1`, `rank_5`, `rank_10`, ... (configurable via `ranks: [..]`)
- `mAP` (mean Average Precision)
- `cmc@1`, `cmc@5`, `cmc@10`, `cmc@20`

### In-training evaluation

Add a `periodic_eval` block to the trainer YAML (see reference above) and the same evaluator runs inline every N epochs, with metrics streamed to TensorBoard under `eval/*`.

## TensorBoard

Enable it once (`logging.tensorboard: true` in the trainer YAML) and every metric the trainer or periodic evaluator computes is persisted as a scalar:

- `loss/train`, `loss/val`
- `lr`
- `train_stats/<key>`, `val_stats/<key>` (running averages of whatever each loss returns)
- `eval/<metric>` from `periodic_eval`

```bash
uv run tensorboard --logdir <checkpoint_save_dir>/runs
```

## Adding a custom component

Adding e.g. a new loss is a four-step process. No core file needs to change.

1. **Implement the class** under `src/loss/my_loss.py`:

   ```python
   from typing import Dict, Tuple
   import torch
   from config.loss.base_loss_config import LossConfig
   from loss.base_loss import BaseLoss

   class MyLoss(BaseLoss):
       def __init__(self, loss_config: LossConfig):
           super().__init__(loss_config)
           self.margin = loss_config.margin     # any YAML key works directly

       def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
           ...
           return loss, {"my_metric": value.item()}
   ```

2. **Register it** in `src/loss/__init__.py`:

   ```python
   from .my_loss import MyLoss
   LOSSES.register("my_loss", MyLoss)
   ```

3. **Drop a YAML** at `configs/loss/my_loss.yaml`:

   ```yaml
   device: cuda
   margin: 0.5
   ```

4. **Point the env at it**:

   ```env
   LOSS=my_loss
   LOSS_CONFIG=./configs/loss/my_loss.yaml
   ```

The exact same recipe applies to backbones, datasets, transformations, samplers, early stoppers, evaluators, and eval datasets — each has its own registry and `Base*` parent class.

## Development

Common tasks:

```bash
uv sync                    # install / refresh the environment
uv run python train.py     # run the trainer (uses .env)
uv run python eval.py      # standalone evaluation (uses env vars)
uv run streamlit run gui/app.py
uv run tensorboard --logdir runs/

uv add <package>           # add a dependency
uv lock --upgrade          # refresh uv.lock
```

The repo deliberately avoids leaf `*Config` classes — every component reads its YAML keys via attribute access on the shared `Base*Config`. Look at `src/loss/triplet_loss.py` and `configs/loss/triplet.yaml` for the canonical pattern.
