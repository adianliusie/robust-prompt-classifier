import random
import os

from typing import List
from types import SimpleNamespace
from tqdm import tqdm
from copy import deepcopy
from functools import lru_cache

from .load_classification_hf import load_hf_cls_data, HF_CLS_DATA
from .load_pair_hf import load_hf_pair_data, HF_PAIR_DATA
from .race import load_mcrc_data, MCRC_DATA

from ..models.tokenizers import load_tokenizer
from ..utils.general import get_base_dir, save_pickle, load_pickle

BASE_PATH = get_base_dir()
BASE_CACHE_PATH = f"{BASE_PATH}/tokenize-cache/"

#== Main DataHandler class ========================================================================#
class DataHandler:
    def __init__(self, trans_name:str, template:str):
        self.trans_name = trans_name
        self.tokenizer = load_tokenizer(trans_name)
        self.template = template

    #== Data processing (i.e. tokenizing text) ====================================================#
    def prep_split(self, data_name:str, mode:str, lim=None):
        split = self.load_split(data_name=data_name, mode=mode, lim=lim)
        data = self._prep_ids(split)
        return data

    @lru_cache(maxsize=10)
    def prep_data(self, data_name, lim=None):
        train, dev, test = self.load_data(data_name=data_name, lim=lim)
        train, dev, test = [self._prep_ids(split) for split in [train, dev, test]]
        return train, dev, test
    
    #== Different tokenization methods for different task set up ==================================#
    def _prep_ids(self, split_data:List[SimpleNamespace]):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            input_text = self._prep_text(ex)
            input_ids = self.tokenizer(input_text).input_ids

            ex.input_text = input_text
            ex.input_ids = input_ids
        return split_data

    def _prep_text(self, ex):
        template = self.template

        # Sentiment Classification template formatting
        if '<t>' in template:
            template = template.replace('<t>',  ex.text)
        if '<t1>' in template or '<t2>' in template:
            template = template.replace('<t1>', ex.text_1)
            template = template.replace('<t2>', ex.text_2)
        if '<m>' in template:
            template = template.replace('<m>',  self.tokenizer.mask_token)

        # MCRC template formatting
        if '<q>' in template:
            template = template.replace('<q>',  ex.question)
        if '<c>' in template:
            template = template.replace('<c>', ex.context)
        if '<o>' in template:
            option_str = ''
            for k, option in enumerate(ex.options):
                char = chr(k + 97) # {0: a, 1:b, 2:c, ...}
                option_str += f"{char}) {option}\n"
            template = template.replace('<o>',  option_str)

        return template
        
    #== Data loading utils ========================================================================#
    @staticmethod
    @lru_cache(maxsize=10)
    def load_data(data_name:str, lim=None):
        if   data_name in HF_CLS_DATA   : train, dev, test = load_hf_cls_data(data_name)
        elif data_name in HF_PAIR_DATA  : train, dev, test = load_hf_pair_data(data_name)
        elif data_name in MCRC_DATA     : train, dev, test = load_mcrc_data(data_name)
        else: raise ValueError(f"invalid dataset name: {data_name}")
          
        train, dev, test = to_namespace(train, dev, test)

        if lim:
            train = rand_select(train, lim)
            dev   = rand_select(dev, lim)
            test  = rand_select(test, lim)
            
        return train, dev, test
    
    @classmethod
    def load_split(cls, data_name:str, mode:str, lim=None):
        split_index = {'train':0, 'dev':1, 'test':2}        
        data = cls.load_data(data_name, lim)[split_index[mode]]
        return data

#== Misc utils functions ============================================================================#
def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]

def to_namespace(*args:List):
    def _to_namespace(data:List[dict])->List[SimpleNamespace]:
        if not hasattr(data[0], 'ex_id'):
            return [SimpleNamespace(ex_id=k, **ex) for k, ex in enumerate(data)]
        else:
            return [SimpleNamespace(**ex) for ex in data]

    output = [_to_namespace(split) for split in args]
    return output if len(args)>1 else output[0]

    