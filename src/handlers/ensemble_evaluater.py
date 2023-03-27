import numpy as np
import os

from .evaluater import Evaluator

class EnsembleEvaluator(Evaluator):
    def __init__(self, exp_path:str, device:str='cuda'):
        self.exp_path = exp_path
        self.paths = [f'{exp_path}/{seed}' for seed in os.listdir(exp_path) if os.path.isdir(f'{exp_path}/{seed}')]
        self.seeds = [Evaluator(seed_path, device) for seed_path in sorted(self.paths)]

    def load_probs(self, data_name:str, mode)->dict:
        seed_probs = [seed.load_probs(data_name, mode) for seed in self.seeds]
        ex_ids = seed_probs[0].keys()
        assert all([i.keys() == ex_ids for i in seed_probs])

        ensemble = {}
        for ex_id in ex_ids:
            probs = [seed[ex_id] for seed in seed_probs]
            probs = np.mean(probs, axis=0)
            ensemble[ex_id] = probs
        return ensemble    
    
    def load_preds(self, data_name:str, mode)->dict:
        probs = self.load_probs(data_name, mode)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds

    def load_layer_h(self, *args, **kwargs):
        seed_1 = self.seeds[0]
        h = seed_1.load_layer_h(*args, **kwargs)
        return h
    