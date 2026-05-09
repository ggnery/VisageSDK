from registry import SAMPLERS

from .base_batch_sampler import BaseBatchSampler
from .facenet_batch_sampler import FacenetBatchSampler

SAMPLERS.register("facenet", FacenetBatchSampler)

__all__ = ["BaseBatchSampler", "FacenetBatchSampler"]
