#!/bin/bash

. /home/server25/anaconda3/etc/profile.d/conda.sh

data_dir=data/sonny/train.mp4

python -c '
from data.dataset.util import construct_stack
import torch
data_dir = "'${data_dir}'"
stack = construct_stack(data_dir, include_audio=False)
torch.save(stack, data_dir + "/mesh_stack.pt")
'