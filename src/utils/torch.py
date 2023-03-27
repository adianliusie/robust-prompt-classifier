import random
import torch
import numpy as np

#== Reproducibility ===============================================================================#
def set_rand_seed(seed_num):
    assert 0 < seed_num <= 10_000, "seed number has to be within 1-10,000"
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

#== Optimisation ==================================================================================#
def build_triangular_scheduler(optimizer, num_warmup_steps:str, num_steps:str):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        return (num_steps - step) / (num_steps - num_warmup_steps)

    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )
    return scheduler