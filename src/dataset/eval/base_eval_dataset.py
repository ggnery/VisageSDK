from PIL import Image
from torch.utils.data import Dataset

from config.dataset.eval.base_eval_dataset_config import EvalDatasetConfig
from transformation.base_transformation import BaseTransformation


class BaseEvalDataset(Dataset):
    """Eval datasets expose a flat `self.data` list of (label, image_path).

    `__getitem__` returns (idx, transformed_image) so evaluators can encode
    each unique image once and look up pairs/groups by index.
    """

    data: list[tuple[str, str]]

    def __init__(self, config: EvalDatasetConfig, transformation: BaseTransformation) -> None:
        super().__init__()
        self.config = config
        self.transform = transformation.transform
        self.data = []

    def __getitem__(self, idx: int):
        _, img_path = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        return idx, self.transform(image)

    def __len__(self) -> int:
        return len(self.data)
