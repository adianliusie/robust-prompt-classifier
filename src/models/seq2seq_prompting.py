import torch
import torch.nn as nn

from types import SimpleNamespace
from typing import List
from transformers import logging
from functools import lru_cache

from .pre_trained_trans import load_seq2seq_transformer
from .tokenizers import load_tokenizer

logging.set_verbosity_error()

class Seq2seqPrompting(torch.nn.Module):
    def __init__(
            self, 
            trans_name:str, 
            label_words:list, 
            decoder_template:str=''
    ):
        super().__init__()
        self.transformer = load_seq2seq_transformer(trans_name)
        self.tokenizer   = load_tokenizer(trans_name)
        self.label_ids   = [int(*self.tokenizer(word, add_special_tokens=False).input_ids) for word in label_words]
        self.decoder_template = decoder_template
        self.device = 'cpu'

    def forward(
        self,
        input_ids,
        attention_mask=None,
    ):
        # set up decoder input ids
        self.decoder_input_ids = self.get_decoder_ids(bsz=input_ids.size(0))

        # encode everything and get MLM probabilities
        trans_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=self.decoder_input_ids
        )
        
        # select MLM probs of the masked positions, only for the label ids
        vocab_logits = trans_output.logits[:,-1]
        class_logits = vocab_logits[:, tuple(self.label_ids)]

        return SimpleNamespace(
            logits=class_logits,
            vocab_logits=vocab_logits
        )

    @lru_cache(maxsize=3)
    def get_decoder_ids(self, bsz) -> List[int]:
        if self.decoder_template:
            # repeat template bsz times
            decoder_input_ids = self.tokenizer(
                [self.decoder_template for _ in range(bsz)], 
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids
            
            # add start token
            decoder_input_ids = self.transformer._shift_right(decoder_input_ids)
        else:
            # set input to start of sentence token
            decoder_input_ids = self.transformer.config.decoder_start_token_id * torch.ones(bsz, 1, dtype=torch.long)
        return decoder_input_ids.to(self.device)

    def update_label_words(self, label_words:str):
        print([self.tokenizer(word, add_special_tokens=False).input_ids for word in label_words])
        self.label_ids = [int(*self.tokenizer(word, add_special_tokens=False).input_ids) for word in label_words]

    def freeze_head(self):
        for param in self.transformer.lm_head.parameters():
            param.requires_grad = False

    def set_prompt_tuning(self):
        pass

    def to(self, device):
        super().to(device)
        
        # update device so decoder_ids on correct device
        self.device = device
        self.get_decoder_ids.cache_clear()