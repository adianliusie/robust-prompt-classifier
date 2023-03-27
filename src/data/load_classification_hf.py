import random
import re

from tqdm import tqdm 
from copy import deepcopy
from typing import List, Dict, Tuple, TypedDict
from datasets import load_dataset
from functools import lru_cache

class SingleText(TypedDict):
    """Output example formatting (only here for documentation)"""
    text : str
    label : int

#== Main loading function =========================================================================# 

HF_CLS_DATA = ['imdb', 'rt', 'sst', 'yelp', 'amazon']
HF_CLS_DATA += [i+'-s' for i in HF_CLS_DATA] # add smaller versions

def load_hf_cls_data(data_name)->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    """ loading sentiment classification datsets available on huggingface hub """
    # if small version needed, split dataset name:
    small = False
    if data_name[-2:] == '-s':
        data_name, _ = data_name.split('-s')
        small = True

    # get the relevant data
    if   data_name == 'imdb':    train, dev, test = load_imdb()
    elif data_name == 'rt':      train, dev, test = load_rotten_tomatoes()
    elif data_name == 'sst':     train, dev, test = load_sst()
    elif data_name == 'yelp':    train, dev, test = load_yelp()
    elif data_name == 'amazon':  train, dev, test = load_amazon()
    else: raise ValueError(f"invalid single text dataset name: {data_name}")

    # if small, then randomly select 5000 points for test
    if small:
        train = rand_select(train, 5000)
        dev   = rand_select(dev, 5000)
        test  = rand_select(test, 5000)   
    return train, dev, test
    
#== sentiment analysis datasets ===================================================================#
def load_imdb()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("imdb")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])
    train, dev, test = _remove_html_tags(train, dev, test)
    return train, dev, test

def load_yelp()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("yelp_polarity")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])
    return train, dev, test

def load_amazon()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("mteb/amazon_polarity")
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['test'])
    return train, dev, test
   
def load_rotten_tomatoes()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset("rotten_tomatoes")
    train = list(dataset['train'])
    dev   = list(dataset['validation'])
    test  = list(dataset['test'])
    return train, dev, test

def load_sst()->Tuple[List[SingleText], List[SingleText], List[SingleText]]:
    dataset = load_dataset('glue', 'sst2')
    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])
    
    train, dev, test = _rename_keys(train, dev, test, old_key='sentence', new_key='text')
    return train, dev, test

#== Util functions for processing data sets =======================================================#
def _create_splits(examples:list, ratio=0.8)->Tuple[list, list]:
    examples = deepcopy(examples)
    split_len = int(ratio*len(examples))
    
    random_seeded = random.Random(1)
    random_seeded.shuffle(examples)
    
    split_1 = examples[:split_len]
    split_2 = examples[split_len:]
    return split_1, split_2

def _rename_keys(train:list, dev:list, test:list, old_key:str, new_key:str):
    train = [_rename_key(ex, old_key, new_key) for ex in train]
    dev   = [_rename_key(ex, old_key, new_key) for ex in dev]
    test  = [_rename_key(ex, old_key, new_key) for ex in test]
    return train, dev, test

def _rename_key(ex:dict, old_key:str='content', new_key:str='text'):
    """ convert key name from the old_key to 'text' """
    ex = ex.copy()
    ex[new_key] = ex.pop(old_key)
    return ex

def _remove_html_tags(train:list, dev:list, test:list):
    train = [_remove_html_tags_ex(ex) for ex in train]
    dev   = [_remove_html_tags_ex(ex) for ex in dev]
    test  = [_remove_html_tags_ex(ex) for ex in test]
    return train, dev, test

def _remove_html_tags_ex(ex:dict):
    CLEANR = re.compile('<.*?>') 
    ex['text'] = re.sub(CLEANR, '', ex['text'])
    return ex
  
def rand_select(data:list, lim:None):
    if data is None: return None
    random_seed = random.Random(1)
    data = data.copy()
    random_seed.shuffle(data)
    return data[:lim]
