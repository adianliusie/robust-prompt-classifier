
 

import torch
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Tuple

from .base import BaseLoss

class Seq2SeqCrossEntropyLoss(BaseLoss):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch: SimpleNamespace) -> Tuple[float, dict]:
        output = self.model(
            input_ids = batch.input_ids, 
            attention_mask = batch.attention_mask, 
            labels = batch.label_ids
        )

        # Cross entropy loss
        loss = output.loss  

        # Masking out all non-labels
        mask = batch.label_ids != -100

        # Token level accuracy
        x = (output.logits.argmax(dim = -1) == batch.label_ids)

        # Masked Token level accuracy
        acc = torch.masked_select(x, mask) 
                
        self.record_metrics({
            'loss': loss.item(),
            'ce': loss.item(),
            'acc': acc.sum() / mask.sum(),
        })

        return SimpleNamespace(
                    loss=loss, 
                    logits=output.logits,
                    model_output=output,
        )
