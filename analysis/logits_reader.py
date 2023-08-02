import numpy as np
import os
import scipy
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import itertools

from collections import defaultdict
from typing import List
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm
from functools import lru_cache
from types import SimpleNamespace

from src.utils.general import load_pickle, load_json
from src.models.seq2seq_prompting import Seq2seqPrompting
from src.models.decoder_prompting import DecoderPrompting
from src.loss.cross_entropy import CrossEntropyLoss
from src.data.data_handler import DataHandler
from src.handlers.batcher import Batcher
from src.models.pre_trained_trans import MLM_TRANSFORMERS, DECODER_TRANSFORMERS, SEQ2SEQ_TRANSFORMERS

class LogitsReader:
    def __init__(self, path, dataset):
        # properties
        self.path = path
        info = load_json(os.path.join(self.path, 'info.json'))
        self.prompt = info['prompt']
        self.all_words = info['words']
        self.model_name = info.get('model', 'flan-t5-large')

        # load labels
        labels = DataHandler.load_labels(dataset)
        self.keys = labels.keys()
        self.labels_np = np.array([labels[ex_id] for ex_id in self.keys])
        if dataset in ['qqp', 'mrpc']:
            print('initial error in qqp- switching labels')
            self.labels_np  = 1 - self.labels_np

        # load logits
        logits_dict = load_pickle(os.path.join(self.path, 'logits.pk'))
        self.logits_np = np.array([logits_dict[ex_id] for ex_id in self.keys])
        self.num_classes = len(set(labels.values()))
        
    #== Methods to load logits and probabilities ========================================================#
    def load_logits(self, label_words:list=None, norm:str=None)->np.ndarray:
        # if words were not used in inference, send error to user
        if not all([w in self.all_words for w in label_words]):
            raise ValueError(
                'Invalid label words: run evaluation with the required label words\n',
                 f"label words: {self.all_words}"
            )
            
        if norm in [None, 'log-norm', 'prob-norm', 'clean-norm', 'null-norm']:
            class_logits, _ = self.norm_class_logits(label_words=label_words, norm=norm)
        
        elif norm == 'balanced':
            class_logits, _ = self.balanced_alpha_logits(label_words=label_words)

        elif norm == 'N-optimal':
            class_logits, _ = self.N_optimal_threshold(label_words=label_words)
            
        elif norm == 'optimal':
            class_logits, _ = self.optimal_threshold(label_words=label_words)
        
        return class_logits

    def load_probs(self, label_words:list=None, norm:bool=False, annealing:bool=False)->np.ndarray:
        logits = self.load_logits(label_words=label_words, norm=norm)
        
        if annealing:
            logits = self.anneal_logits(logits)
        
        probs = scipy.special.softmax(logits, axis=-1)   
        return probs

    #== Methods to evaluate the accuracy and calibration ================================================#
    def calc_acc(self, label_words:list=None, norm:bool=False):
        probs = self.load_probs(label_words=label_words, norm=norm)
        preds = np.argmax(probs, axis=-1)
        acc = sum(preds == self.labels_np)/len(self.labels_np)
        return round(100*acc, 2)
    
    def calc_ECE(self, label_words:list=None, norm:bool=False):
        # get probs and labels
        probs = self.load_probs(label_words=label_words, norm=norm)
        probs_torch = torch.FloatTensor(probs)
        target_torch = torch.LongTensor(self.labels_np)
        
        # calculate ECE
        metric = MulticlassCalibrationError(num_classes=probs.shape[1], n_bins=10, norm='l1')
        output = metric(preds=probs_torch, target=target_torch)
        return 100*float(output)
    
    #== Logits calibration methods =====================================================================#
    def anneal_logits(self, logits):
        pass
    
    def norm_class_logits(self, label_words:list=None, norm:str=None):        
        # get normalising values
        if norm == None:
            norm_values = 0
            
        elif norm == 'log-norm':
            norm_values = np.mean(self.logits_np, axis=0)  
            
        elif norm == 'prob-norm':
            prob_dict = load_pickle(os.path.join(self.path, 'probs.pk'))
            prob_dict_np = np.array([prob_dict[ex_id] for ex_id in self.keys])
            norm_prob_values = np.mean(prob_dict_np, axis=0)
            norm_values = np.log(norm_prob_values)
            
        elif norm == 'clean-norm':
            indices = [self.all_words.index(w) for w in label_words]

            logits = self.logits_np.copy()
            class_logits = logits[:, indices]
            class_probs = scipy.special.softmax(class_logits, axis=-1) 

            avg_clss_probs = np.mean(class_probs, axis=0)  
            norm_values = np.zeros(len(self.all_words))
            norm_values[indices] = np.log(avg_clss_probs)
            
            temp = class_logits - np.log(avg_clss_probs)
            prob_temp = scipy.special.softmax(temp, axis=-1)

        elif norm == 'null-norm':
            norm_values = self.null_values
        
        else:
            raise ValueError(f"invalid normalisation method: {norm}")
                
        # get class logits
        if label_words == None:
            class_logits = None
        else:
            logits = self.logits_np.copy()
            logits -= norm_values
            indices = [self.all_words.index(w) for w in label_words]
            class_logits = logits[:,indices].copy()

        return class_logits, norm_values        

    def balanced_alpha_logits(self, label_words):        
        # get class logits
        logits = self.logits_np.copy()
        indices = [self.all_words.index(w) for w in label_words]
        class_logits = logits[:,indices].copy()
        
        if self.num_classes == 2:
            search_range = np.arange(-20,20,0.1)
        else:
            search_range = np.arange(-15,15,0.5)

        class_ranges = [search_range for i in range(self.num_classes-1)]
        alphas_perm = list(itertools.product(*class_ranges))
        
        # initialise best performance to Null
        best = (None, 1)

        for thresholds in alphas_perm:
            thresh_logits = class_logits.copy()
            # update logits with current threshold
            for k, t in enumerate(thresholds):
                thresh_logits[:, k] -= t
                
            probs = scipy.special.softmax(thresh_logits, axis=-1)   
            avg_class_probs = np.mean(probs, axis=0)
            dist = np.mean(np.abs(avg_class_probs - np.ones_like(avg_class_probs)/self.num_classes))
            if dist < best[1]:
                best = (thresholds, dist)
        
        # update logits accordingly
        thresholds = best[0]
        for k, t in enumerate(thresholds):
            class_logits[:, k] -= t

        return class_logits, best
        
    def optimal_N_logits(self, label_words:list=None):
        norm_values = self.N_optimal_threshold()
        
        # get class logits
        if label_words == None:
            class_logits = None
        else:
            logits = self.logits_np.copy()
            logits -= norm_values
            indices = [self.all_words.index(w) for w in label_words]
            class_logits = logits[:,indices].copy()

        return class_logits, norm_values
    
    @lru_cache(maxsize=10)
    def N_optimal_threshold(self):
        logits = self.logits_np.copy()
        logits_torch = torch.FloatTensor(logits)
        labels_torch = torch.LongTensor(self.labels_np)
    
        biases = torch.nn.Parameter(torch.empty(1, logits_torch.shape[-1]), requires_grad=True)
        torch.nn.init.xavier_uniform_(biases)

        # set up optimization objects
        optimizer = torch.optim.AdamW(
            iter([biases]), 
            lr=1
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        # get label word sets
        label_word_sets = self.get_label_word_sets()
        
        for epoch in tqdm(range(1000)):
            biased_logits = logits_torch - biases
            loss = 0 
            
            index_permutations = list(itertools.product(*label_word_sets))
            for word_set in index_permutations:
                pair_logits = biased_logits[:, word_set]           
                loss += F.cross_entropy(pair_logits, labels_torch)

            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
                        
        norm_values = biases[0].detach().numpy()
        return norm_values
        
    def optimal_threshold(self, label_words):
        """wrapper function to turn input to string (for lru_cache)"""
        joined_label_words = '_'.join(label_words)
        class_logits, best = self._optimal_threshold(joined_label_words)
        return class_logits, best

    @lru_cache(maxsize=10)
    def _optimal_threshold(self, joined_label_words:str):
        label_words = joined_label_words.split('_')
        logits = self.logits_np.copy()
        indices = [self.all_words.index(w) for w in label_words]
        class_logits = logits[:,indices].copy()
        
        if self.num_classes == 2:
            search_range = np.arange(-20,20,0.1)
        else:
            search_range = np.arange(-15,15,0.5)

        class_ranges = [search_range for i in range(self.num_classes-1)]
        alphas_perm = list(itertools.product(*class_ranges))
        
        best = (None, 0)

        for thresholds in alphas_perm:
            # check performance for current threshold
            thresh_logits = class_logits.copy()
            for k, t in enumerate(thresholds):
                thresh_logits[:, k] -= t
            
            pred = np.argmax(thresh_logits, axis=-1)
            acc = sum(pred == self.labels_np)/len(self.labels_np)

            # after the search get best operating performance
            if acc > best[1]:
                best = (thresholds, acc)

        # update logits accordingly
        thresholds = best[0]
        for k, t in enumerate(thresholds):
            class_logits[:, k] -= t

        return class_logits, best
        
    @lru_cache(maxsize=10)
    def _optimal_threshold_old(self, joined_label_words:str):
        label_words = joined_label_words.split('_')
        logits = self.logits_np.copy()
        indices = [self.all_words.index(w) for w in label_words]
        class_logits = logits[:,indices].copy()
        
        if self.num_classes == 2:
            # find best threhsold
            best = (None, 0)
            for t_1 in np.arange(-20,20,0.2):
                thresh_logits = class_logits.copy()
                thresh_logits[:, 1] -= t_1

                pred = np.argmax(thresh_logits, axis=-1)
                acc = sum(pred == self.labels_np)/len(self.labels_np)

                if acc > best[1]:
                    best = (t_1, acc)
            
            # update logits accordingly
            t_1 = best[0]
            class_logits[:, 1] -= t_1

        elif self.num_classes == 3:
            # find best threhsold
            best = (None, 0)
            for t_1 in np.arange(-15,15,0.5):
                for t_2 in np.arange(-15,15,0.5):
                    thresh_logits = class_logits.copy()
                    thresh_logits[:, 1] -= t_1
                    thresh_logits[:, 2] -= t_2

                    pred = np.argmax(thresh_logits, axis=-1)
                    acc = sum(pred == self.labels_np)/len(self.labels_np)

                    if acc > best[1]:
                        best = ((t_1, t_2), acc)
                        
            # update logits accordingly
            t_1, t_2 = best[0]
            class_logits[:, 1] -= t_1
            class_logits[:, 2] -= t_2

        return class_logits, best
    
    @property
    def null_values(self):
        if not hasattr(self, '_null_values'):
            with torch.no_grad():
                # Set up Model and Datahandler
                if self.model_name in SEQ2SEQ_TRANSFORMERS:
                    model = Seq2seqPrompting(trans_name=self.model_name, label_words=self.all_words)
                elif self.model_name in DECODER_TRANSFORMERS:
                    model = DecoderPrompting(trans_name=self.model_name, label_words=self.all_words)
                model_loss = CrossEntropyLoss(model)
                data_handler = DataHandler(trans_name=self.model_name, template=self.prompt)
                batcher = Batcher(max_len=512)

                # crete null example
                """
                ex = SimpleNamespace(
                    ex_id='1',
                    text='text',
                    text_1='text_1',
                    text_2='text_2',
                    label=0
                )
                
                """
                ex = SimpleNamespace(
                    ex_id='',
                    text='',
                    text_1='',
                    text_2='',
                    label=0
                )
                
                # determine output logits for null input
                ex_data = data_handler._prep_ids([ex])
                ex_batch = batcher(data=ex_data, bsz=1, shuffle=False)
                output = model_loss(ex_batch[0])
                base_logits = output.logits.squeeze(0).numpy()

                # cache output
                self._null_values = base_logits
            
        return self._null_values

    #== Plotting Methods ================================================================================#
    def plot_conf_acc_curve(self, label_words:list=None, norm:bool=False, num_bins=10, **kwargs):
        probs = self.load_probs(label_words=label_words, norm=norm)
        preds = np.argmax(probs, axis=-1)

        accuracies = (preds == self.labels_np)
        confidences = np.max(probs, axis=-1)
        
        bins_start = 1/probs.shape[1]
        bins_width = (1 - bins_start)/num_bins
        
        #sorted_points = [(acc, conf) for conf, acc in sorted(zip(confidences, accuracies))]
        
        bins_confs = []
        bins_accs = []
        for i in range(num_bins):
            #bin_min = int(len(sorted_points)*i/num_bins)
            #bin_max = int(len(sorted_points)*(i+1)/num_bins)
            #accs, confs = zip(*sorted_points[bin_min:bin_max])

            bin_min = bins_start + i*bins_width
            bin_max = bins_start + (i+1)*bins_width
            points = [(a, c) for (a, c) in zip(accuracies, confidences) if bin_min<=c<bin_max]
            if points == []:
                continue
            accs, confs = zip(*points)
            
            bins_accs.append(np.mean(accs))
            bins_confs.append(np.mean(confs))
            
        plt.plot(bins_confs, bins_accs, marker='o', **kwargs)
    
    def label_words_box_plot_data(self, prompt_num=None):
        label_word_sets = self.get_label_word_sets()
        output = []
        index_permutations = list(itertools.product(*label_word_sets))
        for word_set in tqdm(index_permutations):
            for norm, name in zip(
                #[None, 'log-norm', 'prob-norm', 'N-optimal', 'optimal'],
                #['standard', 'geometric', 'normalised', 'N-optimal', 'optimal']
                [None, 'null-norm', 'balanced', 'optimal'],
                ['baseline', 'null-norm', 'prior-match', 'optimal']   
                #[None, 'log-norm', 'balanced'],
                #['baseline', 'log-norm', 'balanced']   
            ):
                label_words = [self.all_words[index] for index in word_set]
                acc = self.calc_acc(label_words=label_words, norm=norm)
                
                output.append({
                    'prompt':f"{prompt_num+1}",
                    'probs': name,
                    'acc':acc, 
                }) 
        return output
    
    def threshold_plot(self):
        assert self.num_classes == 2, "only implemented for sentiment classification"

        # loop over all word set permutations
        label_word_sets = self.get_label_word_sets()
        index_permutations = list(itertools.product(*label_word_sets))  
        
        # get all thresholds
        thresholds_norm, thresholds_norm_null, thresholds_opt  = [], [], []
        for word_set in index_permutations:
            i1, i2 = word_set
            label_words = [self.all_words[index] for index in word_set]

            # optimal search threshold
            class_logits, (opt_t, acc) = self.optimal_threshold(label_words)
 
            # normalised_null
            #_, norm_values = self.norm_class_logits(norm='null-norm')
            norm_null_t = self.null_values[i1] - self.null_values[i2]
               
            # clean norm
            #_, norm_values = self.norm_class_logits(label_words=label_words, norm='clean-norm')
            #norm_null_t = norm_values[i2] - norm_values[i1]
    
            # normalised_marg 
            #_, norm_values = self.norm_class_logits(norm='log-norm')
            #norm_t = norm_values[i2] - norm_values[i1]
       
            # balanced
            _, (norm_t, acc) = self.balanced_alpha_logits(label_words=label_words)
             
            thresholds_norm.append(norm_t)
            thresholds_norm_null.append(norm_null_t)
            thresholds_opt.append(opt_t)
        
        #scatter plot
        fig, ax = plt.subplots(figsize=(6,5.5))
            
        ax.scatter(np.exp(thresholds_norm), np.exp(thresholds_opt), color='blue')
        ax.scatter(np.exp(thresholds_norm_null), np.exp(thresholds_opt), color='red')

        # line of best fit
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        
        # format plot
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlabel(r'$\frac{P_{\theta}(w_1|p)}{P_{\theta}(w_2|p)}$', size=18)
        #plt.ylabel(r'$\tau_{optimal}$', size=18)
        #plt.legend(['marg-norm', 'null-norm'])
        plt.xlabel(r'$\alpha$', size=18)
        plt.ylabel(r'$\alpha*$', size=18)
        plt.legend(['prior-match', 'null-norm'])
        
    #== Util Methods ==================================================================================#
    def get_label_word_sets(self)->List[List[int]]:
        num_words = len(self.all_words)
        assert num_words%self.num_classes == 0
        
        words_per_class = int(num_words/self.num_classes)
        word_sets = [list(range(i, i+words_per_class)) for i in range(0, num_words, words_per_class)]
        return word_sets
        