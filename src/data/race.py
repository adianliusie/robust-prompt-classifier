import os
import random

from datasets import load_dataset
from copy import deepcopy
from typing import List, Tuple, TypedDict

from ..utils.general import save_pickle, load_pickle, load_json
from .download import RACE_C_DIR, RECLOR_DIR, download_race_plus_plus, download_reclor
from .load_classification_hf import _create_splits, _rename_keys


class McrcData(TypedDict):
    """Output example formatting (only here for documentation)"""
    ex_id : str
    question : str
    context : str
    options : List[str] 
    label : int

MCRC_DATA = ['race++', 'race', 'race-M', 'race-H', 'race-C', 'reclor', 'cosmos', 'race-ctx-rand']
def load_mcrc_data(data_name)->Tuple[List[dict], List[dict], List[dict]]:
    """ loading multiple choice question answering datasets """
    # get the relevant data
    if   data_name == 'race++': train, dev, test = load_race(levels=['M', 'H', 'C'])
    elif data_name == 'race':   train, dev, test = load_race(levels=['M', 'H'])
    elif data_name == 'race-M': train, dev, test = load_race(levels=['M'])
    elif data_name == 'race-H': train, dev, test = load_race(levels=['H'])
    elif data_name == 'race-C': train, dev, test = load_race(levels=['C'])
    elif data_name == 'reclor': train, dev, test = load_reclor()
    elif data_name == 'cosmos': train, dev, test = load_cosmos()
    elif data_name == 'race-ctx-rand': train, dev, test = load_random_context_race(levels=['M', 'H', 'C'])
    else: raise ValueError(f"invalid single text dataset name: {data_name}")
    return train, dev, test
    
#== RACE ==========================================================================================#
def load_race(levels=['M', 'H', 'C'])->List[dict]:
    #load RACE-M and RACE-H data from hugginface
    race_data = {}
    if 'M' in levels: race_data['M'] = load_dataset("race", "middle")
    if 'H' in levels: race_data['H'] = load_dataset("race", "high")
    if 'C' in levels: race_data['C'] = load_race_c()

    #load and format each split, for each difficulty level, and add to data
    SPLITS = ['train', 'validation', 'test']
    train_all, dev_all, test_all = [], [], []
    for char, difficulty_data in race_data.items():
        train = difficulty_data['train']
        dev   = difficulty_data['validation']
        test  = difficulty_data['test']

        # save ex_id (with difficulty level)
        for split in train, dev, test:
            for k, ex in enumerate(split):
                ex['ex_id'] = f'{char}_{k}'

        # append to the overall output splits
        train_all += train
        dev_all   += dev
        test_all  += test

    train_all, dev_all, test_all = _rename_keys(train_all, dev_all, test_all, old_key='article', new_key='context')
    train_all, dev_all, test_all = _rename_keys(train_all, dev_all, test_all, old_key='answer', new_key='label')
    return train_all, dev_all, test_all

def load_race_c():    
    # Download data if missing
    if not os.path.isdir(RACE_C_DIR):
        download_race_plus_plus()
    
    # Load cached data if exists, otherwise process and cache
    pickle_path = os.path.join(RACE_C_DIR, 'cache.pkl')    
    if os.path.isfile(pickle_path):
        train, dev, test = load_pickle(pickle_path)
    else:
        splits_path = [f'{RACE_C_DIR}/{split}' for split in ['train', 'dev', 'test']]
        train, dev, test = [load_race_c_split(path) for path in splits_path]
        save_pickle(data=[train, dev, test], path=pickle_path)
    return {'train':train, 'validation':dev, 'test':test}

def load_race_c_split(split_path:str):
    file_paths = [f'{split_path}/{f_path}' for f_path in os.listdir(split_path)]
    outputs = []
    for file_path in file_paths:
        outputs += load_race_file(file_path)
    return outputs

def load_race_file(path:str):
    file_data = load_json(path)
    article = file_data['article']
    answers = file_data['answers']
    options = file_data['options']
    questions = file_data['questions']
    
    outputs = []
    assert len(questions) == len(options) == len(answers)
    for k in range(len(questions)):
        ex = {'question':questions[k], 
              'article':article,
              'options':options[k],
              'answer':answers[k]}
        outputs.append(ex)
    return outputs

def load_random_context_race(levels=['M', 'H', 'C']):
    train, dev, test = load_race(levels)
    train, dev, test = [shuffle_context(split) for split in [train, dev, test]]
    return train, dev, test

#== COSMOS ========================================================================================#
def load_cosmos():
    #load RACE-M and RACE-H data from hugginface
    dataset = load_dataset("cosmos_qa")

    train_data = list(dataset['train'])
    train, dev = _create_splits(train_data, 0.8)
    test       = list(dataset['validation'])

    # format the options to be consistent
    for split in train, dev, test:
        for ex in split:
            ex['options'] = [ex[f'answer{k}'] for k in [0,1,2,3]]

    train, dev, test = _rename_keys(train, dev, test, old_key='id', new_key='ex_id')
    return train, dev, test

#== RECLOR ========================================================================================#
def load_reclor():    
    # download data if missing
    if not os.path.isdir(RECLOR_DIR):
        download_reclor()
    
    # load and prepare each data split
    splits_path = [f'{RECLOR_DIR}/{split}.json' for split in ['train', 'val', 'val']]
    train, dev, test = [load_json(path) for path in splits_path]

    train, dev, test = _rename_keys(train, dev, test, old_key='answers', new_key='options')
    train, dev, test = _rename_keys(train, dev, test, old_key='id_string', new_key='ex_id')
    return train, dev, test

def load_random_context_reclor():
    train, dev, test = load_reclor()
    train, dev, test = [shuffle_context(split) for split in [train, dev, test]]
    return train, dev, test

#== UTILS =========================================================================================#
def shuffle_context(data):
    #get all contexts in dataset and shuffle
    contexts = [ex['context'] for ex in data]
    random.seed(1)
    random.shuffle(contexts)

    # change contexts for each sample
    data = deepcopy(data)
    for k, ex in enumerate(data):
        ex['context'] = contexts[k]
    return data