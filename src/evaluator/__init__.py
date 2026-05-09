from registry import EVALUATORS

from .base_evaluator import BaseEvaluator
from .verification_evaluator import VerificationEvaluator
from .identification_evaluator import IdentificationEvaluator

EVALUATORS.register("verification", VerificationEvaluator)
EVALUATORS.register("identification", IdentificationEvaluator)

__all__ = ["BaseEvaluator", "VerificationEvaluator", "IdentificationEvaluator"]
