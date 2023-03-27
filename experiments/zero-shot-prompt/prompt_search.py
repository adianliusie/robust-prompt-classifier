
import os
import numpy as np
import torch
import torch.nn.functional as F
import itertools

from collections import defaultdict

from copy import deepcopy
from tqdm import tqdm
from typing import List

from src.handlers.trainer import Trainer
from src.handlers.evaluater import Evaluater
from src.utils.general import save_pickle, save_json
from src.utils.parser import get_model_parser, get_train_parser
from src.utils.analysis import probs_to_preds


def prompt_search_experiment(
    datasets:List[str], 
    templates:List[str], 
    label_word_sets:List[List[str]], 
    save_probs:bool=False
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
    os.makedirs(output_path)
    save_json(templates, os.path.join(output_path, 'prompts.json'))                
    save_json(label_word_sets, os.path.join(output_path, 'label_words.json'))                

    # run analysis in the entire domain
    with torch.no_grad():
        for dataset in datasets:
            print(dataset)
            for temp_num, template in enumerate(templates):
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
                probabilities = defaultdict(dict)
                total_logits = np.zeros(len(label_words))

                for batch in tqdm(eval_batches):
                    ex_id = batch.ex_id[0]
                    output = evaluater.model_loss(batch)

                    logits = output.logits.squeeze(0)
                    total_logits += logits.cpu().numpy()

                    # consider all possible label word combinations
                    label_words_permutations = list(itertools.product(*label_word_sets))
                    for word_set in label_words_permutations:
                        indices = (word_to_idx[word] for word in word_set)

                        prob = F.softmax(logits[tuple(indices),], dim=-1)
                        probabilities['_'.join(word_set)][ex_id] = prob.cpu().numpy()

                # create results path
                dataset_prompt_path = os.path.join(output_path, dataset, f"prompt_{temp_num}")
                os.makedirs(dataset_prompt_path)
                os.makedirs(os.path.join(dataset_prompt_path, 'probs'))

                print(dataset_prompt_path, '\n\n\n')
                # run evaluation for the prompt template and task
                accuracies = {}
                labels = evaluater.load_labels(dataset, 'test', lim=lim)
                for words, probs in probabilities.items():
                    preds = probs_to_preds(probs)
                    if dataset == 'hans' or dataset == 'hans-s':
                        preds = {k:(0 if v==0 else 1) for k, v in preds.items()}
                    acc = evaluater.calc_acc(preds, labels)
                    accuracies[words] = acc

                    # save probabilities
                    if save_probs:
                        save_pickle(probs, os.path.join(dataset_prompt_path, 'probs', f"{words}.pk"))                

                # save accuracies
                save_json(accuracies, os.path.join(dataset_prompt_path, 'performance.json'))

                # print accuracies
                print(template)
                for words, acc in accuracies.items():
                    print(f"{dataset:<15}  {words:<15}  {acc:.2f}")
                print('\n')

                # save average logits
                avg_logits = total_logits/len(eval_data)
                logits_dict = {word: float(l) for word, l in zip(label_words, avg_logits)}
                save_json(logits_dict, os.path.join(dataset_prompt_path, 'logits.json'))

                #print average logits of each label word
                print(''.join(f"{word:>10}" for word in label_words))
                print(''.join(f"{round(v, 3):>10}" for v in avg_logits))
