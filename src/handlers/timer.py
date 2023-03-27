import torch
import logging
import time

from collections import namedtuple
from .trainer import Trainer

# Create Logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


 #== Main Training Methods =====================================================================#
class Timer(Trainer):
    def time(self, args: namedtuple):
        self.save_args('train_args.json', args)
 
        # set up optimization objects
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr)

        # set up model
        self.to(args.device)
        self.model.train()
        self.log_num_params()

        # Reset loss metrics
        self.best_dev = (0, {})
        self.model_loss.reset_metrics()

        # Setup wandb for online tracking of experiments
        if args.wandb: self.setup_wandb(args)

        for epoch in range(1, args.epochs+1):
            start_time = time.time()
            
            # freeze transformer for first couple of epochs if set
            if args.freeze_trans:
                if epoch <= args.freeze_trans: self.model.freeze_transformer()
                else: self.model.unfreeze_transformer()

            #== Training =============================================
            train_batches = self.batcher.dummy_batches(
                seq_len = args.seq_len,
                data_len = args.data_len,
                bsz = args.bsz, 
                decoder_len = args.decoder_len
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

                    
            #== Print Epoch Timing ============================================
            end_time = time.time()
            epoch_time = end_time-start_time
            logger.info(f"#== EPOCH TIME: {epoch_time:.2f} ======================#")

            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            print(f"Total: {t/10**9:.1f},  Allocated + Reserved: {(a + r)/10**9:.1f}")
