#!/bin/bash

# activate environment
source /home/al826/rds/hpc-work/envs/env_1/bin/activate

#load any enviornment variables needed
source ~/.bashrc
TOKENIZERS_PARALLELISM=false

# cache command 
echo $@ >> CMDs

python ../train.py $@
