from typing import override
from config.batch_sampler.base_batch_sampler_config import BaseBatchSamplerConfig

class FacenetBatchSamplerConfig(BaseBatchSamplerConfig):
    faces_per_identity: int
    num_identities_per_batch: int
    
    @override
    def build_config(self) -> None:
        self.faces_per_identity = self.config["faces_per_identity"]
        self.num_identities_per_batch = self.config["num_identities_per_batch"]