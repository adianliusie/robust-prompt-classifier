import os
import numpy as np
import torch
import torch.nn.functional as F
import itertools

from collections import defaultdict

from copy import deepcopy
from tqdm import tqdm
from typing import List
from types import SimpleNamespace

from src.handlers.trainer import Trainer
from src.handlers.evaluater import Evaluater
from src.utils.general import save_pickle, save_json
from src.utils.parser import get_model_parser, get_train_parser
from src.utils.analysis import probs_to_preds


def prompt_search_experiment(
    datasets:List[str], 
    templates:List[str], 
    label_word_sets:List[List[str]], 
):
    #== Parser ====================================================================================#
    model_parser = get_model_parser()
    train_parser = get_train_parser()

    # Parse system input arguments 
    model_args, moargs = model_parser.parse_known_args()
    train_args, toargs = train_parser.parse_known_args()
    
    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"
    
    # get experiment specific arguments
    lim = train_args.lim 
    output_path = model_args.path

    #== Set Up Zero Shot Model ====================================================================#
    trainer = Trainer(f'models/{model_args.transformer}', model_args)
    train_args.lim = 0
    trainer.train(train_args)

    #== Set Up Evaluation =========================================================================#
    evaluater = deepcopy(trainer)
    evaluater.__class__ = Evaluater
    evaluater.device = 'cuda'
    evaluater.model.eval()

    # update model rods used to get logits
    label_words = [word for label_class in label_word_sets for word in label_class]
    evaluater.model.update_label_words(label_words)
    word_to_idx = {word:k for k, word in enumerate(label_words)}

    # save prompts used in the experiment
    if not os.path.isdir(output_path):
        os.makedirs(output_path)            

    # run analysis in the entire domain
    with torch.no_grad():
        for dataset in datasets:
            print(dataset)
            for temp_num, template in enumerate(templates):
                print(temp_num)
                # set the prompt template
                evaluater.data_handler.template = template
                evaluater.data_handler.prep_data.cache_clear()

                # get the evaluation batches
                eval_data = evaluater.data_handler.prep_split(dataset, 'test', lim=lim)
                eval_batches = evaluater.batcher(
                    data = eval_data, 
                    bsz = 1, 
                    shuffle = False
                )  

                # container for outputs
                logits_dict = defaultdict(dict)
                raw_probs_dict = defaultdict(dict)
                
                # go through every example
                for batch in tqdm(eval_batches):
                    ex_id = batch.ex_id[0]
                    output = evaluater.model_loss(batch)
                    
                    logits = output.logits.squeeze(0)
                    vocab_probs = output.model_output.raw_class_probs.squeeze(0)

                    #save details
                    logits_dict[ex_id] = logits.cpu().numpy()
                    raw_probs_dict[ex_id] = vocab_probs.cpu().numpy()

                # create results path
                dataset_prompt_path = os.path.join(output_path, dataset, f"prompt_{temp_num}")
                os.makedirs(dataset_prompt_path)

                # save all logits per example
                save_pickle(logits_dict, os.path.join(dataset_prompt_path, 'logits.pk'))                

                # save all probabilities per examples
                save_pickle(raw_probs_dict, os.path.join(dataset_prompt_path, 'probs.pk'))                

                # save prompt and labels words:
                info = {
                    'prompt':template, 
                    'words':label_words, 
                    'word_sets':label_word_sets,
                    'model':model_args.transformer
                }
                save_json(info, os.path.join(dataset_prompt_path, 'info.json'))                