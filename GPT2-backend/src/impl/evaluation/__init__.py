from typing import Dict
import torch

from src.interfaces.evaluation import IEvaluation
from src.interfaces.model import IModel
from src.impl.evaluation.loss import estimate_loss

class Evaluation(IEvaluation):
    def __init__(self, model: IModel):
        super().__init__(model)

    def estimate_loss(self, data: torch.Tensor) -> Dict[str, float]:
        return estimate_loss(self.model, data)

    def evaluate_metrics(self, data: torch.Tensor) -> Dict[str, float]:
        pass