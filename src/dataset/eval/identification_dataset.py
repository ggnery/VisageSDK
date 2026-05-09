"""Identification dataset: gallery + probe sets in ImageFolder layout.

Layout:
    <eval_dir>/gallery/<person>/*.jpg
    <eval_dir>/probe/<person>/*.jpg

The dataset stores all images flat with a `roles` array indicating
"gallery" vs "probe" so an evaluator can split them after encoding.
"""

from pathlib import Path
from typing import List

from typing_extensions import override

from config.dataset.eval.base_eval_dataset_config import EvalDatasetConfig
from dataset.eval.base_eval_dataset import BaseEvalDataset
from transformation.base_transformation import BaseTransformation


_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _scan(split_dir: Path) -> List[tuple]:
    pairs: List[tuple] = []
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in _VALID_EXTS:
                pairs.append((cls_dir.name, str(img.absolute())))
    return pairs


class IdentificationDataset(BaseEvalDataset):
    """Stores gallery and probe images. `roles[i]` ∈ {"gallery", "probe"}."""

    roles: List[str]

    @override
    def __init__(self, config: EvalDatasetConfig, transformation: BaseTransformation) -> None:
        super().__init__(config, transformation)

        eval_dir = Path(config.eval_dir)
        gallery_dir = eval_dir / "gallery"
        probe_dir = eval_dir / "probe"
        if not gallery_dir.exists():
            raise FileNotFoundError(f"Gallery dir not found: {gallery_dir}")
        if not probe_dir.exists():
            raise FileNotFoundError(f"Probe dir not found: {probe_dir}")

        self.roles = []
        for entry in _scan(gallery_dir):
            self.data.append(entry)
            self.roles.append("gallery")
        for entry in _scan(probe_dir):
            self.data.append(entry)
            self.roles.append("probe")
