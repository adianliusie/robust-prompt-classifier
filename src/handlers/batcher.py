import torch
import random

from itertools import islice
from typing import List
from types import SimpleNamespace

class Batcher:
    def __init__(self, max_len:int):
        self.device     = torch.device('cpu')
        self.max_len    = max_len

    def batches(self, data:list, bsz:int, shuffle:bool=False, data_ordering=False):
        """splits the data into batches and returns them"""
        examples = self._prep_examples(data)
        if data_ordering: examples.sort(key=lambda x: len(x.input_ids))
        elif shuffle: random.shuffle(examples)
        batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
        if shuffle and data_ordering: random.shuffle(batches)
        return [self.batchify(batch) for batch in batches]
  
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ex_id, input_ids, mask_positions, labels = zip(*batch)  
        input_ids, attention_mask = self._get_padded_ids(input_ids)
        labels = torch.LongTensor(labels).to(self.device)

        return SimpleNamespace(
            ex_id=ex_id, 
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        )

    def _prep_examples(self, data:list):
        """ sequence classification input data preparation"""
        prepped_examples = []
        for ex in data:
            ex_id     = ex.ex_id
            input_ids = ex.input_ids
            label     = ex.label
            mask_position = getattr(ex, 'mask_position', None)

            if self.max_len and len(input_ids) > self.max_len:   
                # if truncate, must move mask_position (assumed to be in the final 512 tokens)         
                if mask_position:
                    mask_position = mask_position - len(input_ids) + 512
                input_ids = [input_ids[0]] + input_ids[-(self.max_len-1):]

            prepped_examples.append([ex_id, input_ids, mask_position, label])
                
        return prepped_examples
                
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
    
    def _get_padded_ids(self, ids:list, pad_id=0)->List[torch.LongTensor]:
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [pad_id]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask

    def __call__(self, *args, **kwargs):
        """routes the main method do the batches function"""
        return self.batches(*args, **kwargs)
    
    #== Debugging Tool to measure Computation time required ===============
    def dummy_batches(self, seq_len, data_len:int=10_000, bsz:int=4, decoder_len=None):
        def random_batch():
            input_ids = torch.randint(low=0, high=1000, size=(bsz, seq_len), dtype=torch.long).to(self.device)
            attention_mask = torch.ones(bsz, seq_len, dtype=torch.long).to(self.device)
            labels =  torch.randint(low=0, high=1, size=(bsz,), dtype=torch.long).to(self.device)
            
            return SimpleNamespace(
                ex_id=0, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )

        return [random_batch() for _ in range(int(data_len // bsz))]