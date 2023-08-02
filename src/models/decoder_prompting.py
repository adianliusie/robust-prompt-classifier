import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
from typing import List
from transformers import logging
from functools import lru_cache

from .pre_trained_trans import load_decoder_transformer
from .tokenizers import load_tokenizer

class DecoderPrompting(torch.nn.Module):
    def __init__(
            self, 
            trans_name:str, 
            label_words:list, 
    ):
        super().__init__()
        self.transformer = load_decoder_transformer(trans_name)
        self.tokenizer   = load_tokenizer(trans_name)
        label_ids = [self.tokenizer(word, add_special_tokens=False).input_ids for word in label_words]
        if any([len(i)>1 for i in label_ids]):
            print('warning: some label words are tokenized to multiple words')
        
        self.label_ids   = [int(self.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]
        self.device = 'cpu'

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        # encode everything and get MLM probabilities
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # select MLM probs of the masked positions, only for the label ids
        vocab_logits = trans_output.logits[:,-1]
        class_logits = vocab_logits[:, tuple(self.label_ids)]
        raw_class_probs = F.softmax(vocab_logits, dim=-1)[:, tuple(self.label_ids)]

        return SimpleNamespace(
            logits=class_logits,
            vocab_logits=vocab_logits,
            raw_class_probs=raw_class_probs
        )

    def update_label_words(self, label_words:str):
        print([self.tokenizer(word, add_special_tokens=False).input_ids for word in label_words])
        self.label_ids = [int(self.tokenizer(word, add_special_tokens=False).input_ids[0]) for word in label_words]

    def freeze_head(self):
        for param in self.transformer.lm_head.parameters():
            param.requires_grad = False

    def to(self, device):
        super().to(device)
        
        # update device so decoder_ids on correct device
        self.device = device