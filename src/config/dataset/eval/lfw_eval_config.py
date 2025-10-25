from config.dataset.eval.base_eval_dataset_config import BaseEvalDatasetConfig

class LFWEvalDatasetConfig(BaseEvalDatasetConfig):
    pairs_path: str

    def build_config(self) -> None:
        self.pairs_path = self.config["pairs_path"]
        