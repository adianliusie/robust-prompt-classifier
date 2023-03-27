import torch
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Tuple

from .base import BaseLoss

class CrossEntropyLoss(BaseLoss):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:
        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
        )

        # Cross entropy loss
        logits = output.logits
        loss = F.cross_entropy(logits, batch.labels)

        # Masking out all non-labels
        hits = torch.argmax(output.logits, dim=-1) == batch.labels
        acc = hits.sum()/len(batch.labels)

        #record training metrics
        self.record_metrics({
            'loss': loss.item(),
            'acc': acc.item(),
            'select': acc.item()
        })

        return SimpleNamespace(
                    loss=loss, 
                    logits=logits, 
        )


