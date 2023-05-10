import os
import logging
import wandb
import torch
import re
import random

from collections import namedtuple
from types import SimpleNamespace
from typing import Optional
from tqdm import tqdm
from copy import deepcopy

from .batcher import Batcher
from ..data.data_handler import DataHandler
from ..models.pre_trained_trans import SEQ2SEQ_TRANSFORMERS, MLM_TRANSFORMERS
from ..models.mlm_prompting import MlmPrompting
from ..models.seq2seq_prompting import Seq2seqPrompting
from ..utils.general import save_json, load_json
from ..utils.torch import set_rand_seed
from ..loss.cross_entropy import CrossEntropyLoss

# Create Logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer(object):
    """ Base class for finetuning transformer to datasets """
    def __init__(self, path: str, args: namedtuple):
        self.setup_exp(path, args)
        self.setup_helpers(args)
        self.log_num_params()

    def setup_helpers(self, args: namedtuple):
        # quick checks to ensure arguments are valid
        assert len(args.label_words) == args.num_classes, "need a label word for each class"

        # set up attributes 
        self.model_args = args
        self.data_handler = DataHandler(trans_name=args.transformer, template=args.template)
        self.batcher = Batcher(max_len=args.maxlen)

        # set up model
        if args.transformer in MLM_TRANSFORMERS:
            self.model = MlmPrompting(trans_name=args.transformer, label_words=args.label_words)
        elif args.transformer in SEQ2SEQ_TRANSFORMERS:
            self.model = Seq2seqPrompting(trans_name=args.transformer, label_words=args.label_words)

        # select the loss function
        self.model_loss = CrossEntropyLoss(self.model)

    #== Main Training Methods =====================================================================#
    def train(self, args: namedtuple):
        optimizer = self.set_up_train(args)

        # if zero shot break
        if args.lim == 0 and args.lim is not None:
            self.save_model()
            return 
        
        # Get train and val split of data
        train, dev, test = self.data_handler.prep_data(
            data_name=args.dataset, 
            lim=args.lim
        )

        for epoch in range(1, args.epochs+1):
            #== Training =============================================
            train_batches = self.batcher(
                data = train, 
                bsz = args.bsz, 
                shuffle = True,
                data_ordering = getattr(args, 'data_ordering', False)
            )

            for step, batch in enumerate(train_batches, start = 1):
                output = self.model_loss(batch)
                optimizer.zero_grad()
                output.loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clip)
                optimizer.step()
        
                # Print train performance every log_every samples
                if step % (args.log_every//args.bsz) == 0:
                    metrics = self.get_metrics()
                    
                    self.log_metrics(
                        metrics = metrics,
                        mode = 'train', 
                        epoch = epoch,
                        ex_step = step*args.bsz
                    )

                    if args.wandb: self.log_wandb(metrics, mode='train')
                    self.model_loss.reset_metrics()   

                # if val_every given then run validation within epoch
                if step % (args.val_every//args.bsz) == 0:
                    self.validate(dev, epoch, ex_step=step*args.bsz, wandb=args.wandb)

            #== Validation ============================================
            self.validate(dev, epoch, ex_step='end', wandb=args.wandb)

            best_epoch = int(self.best_dev[0].split('-')[0])
            if (args.early_stop) and (epoch - best_epoch >= args.early_stop):
                break

        #== Retraining for Frozen + Prompt-finetuning
        if args.freeze_trans == 'probe-finetune':
            self.probe_finetune_retrain()

    def set_up_train(self, args: namedtuple):
        self.save_args('train_args.json', args)
 
        # save model at the start, in case never better
        self.save_model()
            
        # set up optimization objects
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr)

        # set up model
        self.to(args.device)
        self.model.train()

        # Reset loss metrics
        self.best_dev = ('0-start', {})
        self.model_loss.reset_metrics()

        # Setup wandb for online tracking of experiments
        if args.wandb: self.setup_wandb(args)

        # freeze transformer if selected
        if args.freeze_trans in ['probe-finetune', 'freeze']:
            self.model.freeze_transformer()
        elif args.freeze_trans == 'head':
            self.model.freeze_head()
        
        return optimizer
    
    def probe_finetune_retrain(self, args):
        # load best linear probing model
        self.load_model()
        
        # overwrite args to not freeze layers
        self.model.unfreeze_transformer()
        args2 = deepcopy(args)
        setattr(args2, 'freeze_trans', None)
        
        # train model with finetuning
        self.train(args2)

        # save original args
        self.save_args('train_args.json', args)

    def validate(self, dev, epoch:int, ex_step:int=None, wandb=False):
        metrics = self.run_validation(dev, mode='dev')
        self.log_metrics(metrics = metrics, mode='dev')
        if wandb: self.log_wandb(metrics, mode= 'dev')

        if metrics['loss'] < self.best_dev[1].get('loss', 100):
            self.best_dev = (f'{epoch}-{ex_step}', metrics.copy())
            self.save_model()
        
        self.log_metrics(metrics=self.best_dev[1], mode='dev-best', ex_step=self.best_dev[0])
        return metrics 
    
    @torch.no_grad()
    def run_validation(self, data, bsz:int=1, mode='dev'):
        self.model.eval()
        self.model_loss.reset_metrics()

        val_batches = self.batcher(
            data = data, 
            bsz = bsz, 
            shuffle = False
        )

        for batch in val_batches:
            self.model_loss.eval_forward(batch)
        
        metrics = self.get_metrics()
        self.model.train()
        return metrics

    #== Logging Utils =============================================================================#
    def get_metrics(self):
        metrics = {key: value.avg for key, value in self.model_loss.metrics.items()}
        return metrics

    def log_metrics(self, metrics: dict, mode: str, epoch:str=None, ex_step:int=None):
        # Create logging header
        if   mode == 'train'        : msg = f"epoch {epoch:<2}   ex {ex_step:<7} "
        elif mode in ['dev', 'test']: msg = f"{mode:<10}" + 12*' '
        elif mode == 'dev-best'     : msg = f"best-dev {'(' + str(ex_step) + ')':<12} "    
        else: raise ValueError()

        # Get values from Meter and print all
        for key, value in metrics.items():
            msg += f'{key}: {value:.3f}  '
        
        # Log Performance 
        logger.info(msg)

    def log_wandb(self, metrics, mode):
        if mode != 'train': 
            metrics = {f'{mode}-{key}': value for key, value in metrics.items()}
        wandb.log(metrics)

    #== Saving Utils ==============================================================================#
    def save_args(self, name: str, data: namedtuple):
        """ Saves arguments into json format """
        path = os.path.join(self.exp_path, name)
        save_json(data.__dict__, path)

    def load_args(self, name: str) -> SimpleNamespace:
        path = os.path.join(self.exp_path, name)
        args = load_json(path)
        return SimpleNamespace(**args)
    
    def save_model(self, name : str ='model'):
        # Get current model device
        device = next(self.model.parameters()).device
        
        # Save model in cpu
        self.model.to("cpu")
        path = os.path.join(self.exp_path, 'models', f'{name}.pt')
        torch.save(self.model.state_dict(), path)

        # Return to original device
        self.model.to(device)

    def load_model(self, name: str = 'model'):
        name = name if name is not None else 'model'
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.exp_path, 'models', f'{name}.pt')
            )
        )

    #== Experiment Utils ==========================================================================#
    def setup_exp(self, exp_path: str, args: namedtuple):
        self.exp_path = exp_path

        # prepare experiment directory structure
        if not os.path.isdir(self.exp_path):
            os.makedirs(self.exp_path)

        mod_path = os.path.join(self.exp_path, 'models')
        if not os.path.isdir(mod_path):
            os.makedirs(mod_path)

        eval_path = os.path.join(self.exp_path, 'eval')
        if not os.path.isdir(eval_path):
            os.makedirs(eval_path)

        # add file handler to logging
        fh = logging.FileHandler(os.path.join(exp_path, 'train.log'))
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(fh)

        # set random seed
        if (args.rand_seed is None) and ('/seed-' in exp_path):
            args.rand_seed = int(exp_path.split('/seed-')[-1])        
        elif args.rand_seed is None:
            args.rand_seed = random.randint(1,1000)
        set_rand_seed(args.rand_seed)
        logger.info(f'random seed set to {args.rand_seed}')
        
        #save model arguments
        self.save_args('model_args.json', args)

    def to(self, device):
        assert all([hasattr(self, i) for i in ['model', 'batcher', 'model_loss']]) 
        self.model.to(device)
        self.batcher.to(device)
        self.model_loss.to(device)

    def setup_wandb(self, args: namedtuple):
        # remove everything before */trained_models for exp_name
        exp_name = re.sub(r'^.*?trained_models', '', self.exp_path)

        # remove the final -seed-i from the group name
        group_name = '/seed'.join(exp_name.split('/seed')[:-1])

        #init wandb project
        wandb.init(
            project=f"data-pruning-{args.dataset}",
            entity='adian',
            name=exp_name, 
            group=group_name,
            dir=self.exp_path
        )

        # save experiment config details
        cfg = {
            'dataset': args.dataset,
            'bsz': args.bsz,
            'lr': args.lr,
            'transformer': self.model_args.transformer,
        }

        wandb.config.update(cfg) 
        wandb.watch(self.model)
        
    def log_num_params(self):
        """ prints number of paramers in model """
        logger.info("Number of parameters in model {:.1f}M".format(
            sum(p.numel() for p in self.model.parameters()) / 1e6
        ))


