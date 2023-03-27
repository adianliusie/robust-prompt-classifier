import torch

from types import SimpleNamespace
from transformers import logging

from .pre_trained_trans import load_MLM_transformer
from .tokenizers import load_tokenizer

logging.set_verbosity_error()

class MlmPrompting(torch.nn.Module):
    def __init__(
            self, 
            trans_name:str, 
            label_words:list, 
    ):
        super().__init__()
        self.transformer = load_MLM_transformer(trans_name)
        self.tokenizer   = load_tokenizer(trans_name)
        self.label_ids   = [self.tokenizer(word).input_ids[1] for word in label_words]
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        # encode everything and get MLM probabilities
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # select MLM probs of the masked positions, only for the label ids
        mask_pos_logits = trans_output.logits[input_ids == self.tokenizer.mask_token_id]
        class_logits = mask_pos_logits[:, tuple(self.label_ids)]
        #h = trans_output.hidden_states[-1][input_ids == self.tokenizer.mask_token_id]

        return SimpleNamespace(
            #h = h, 
            logits=class_logits,
            vocab_logits=mask_pos_logits
        )
    
    def update_label_words(self, label_words:str):
        self.label_ids = [self.tokenizer(word).input_ids[1] for word in label_words]

    def freeze_head(self):
        for param in self.transformer.lm_head.parameters():
            param.requires_grad = False

