"""
Code modified based on the following repos:
 - FNO: https://github.com/zongyi-li/fourier_neural_operator
 - UNO: https://github.com/ashiq24/UNO
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import distributed_utils, data_utils, training_utils

from timeit import default_timer
from models.model_utils import count_params
from pathlib import Path

import os
import glob
import pickle
import importlib

################################################################
# Distributed setup
################################################################

world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
local_rank = distributed_utils.setup(rank, world_size)


################################################################
# load data
################################################################
batch_size = 8
n_workers = 8
data_path = '/p/gpfs1/kong11/data/SMART/data_assimilation_toolbox_test_data/simulation_data/normalized_training'


print("### Loading data ...")
t1 = default_timer()
train_files = glob.glob(os.path.join(data_path, 'training/*.npz'))
train_loader = data_utils.load_data(files=train_files,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    n_workers=n_workers,
                                    divide='Train')

val_files = glob.glob(os.path.join(data_path, 'validation/*.npz'))
val_loader = data_utils.load_data(files=val_files,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    n_workers=n_workers,
                                    divide='Validation')

ntrain = len(train_loader.dataset)
nval = len(val_loader.dataset)

t2 = default_timer()
print(f"It takes {t2 - t1}s to load {ntrain} training data and {nval} validation data")

################################################################
# load model
################################################################

model_name = 'uno_3d'
model_mod = importlib.import_module(f"models.{model_name}")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

width = 16
in_channels = 3
epochs = 500
learning_rate = 0.1

checkpoint_dir = f'../output/{model_name}_10nodes_r4_g1_model2-lR_scheduler_0.1/'
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

print("### Loading model ...")

if model_name == 'fno_3d':
    mode = 20
    model = model_mod.FNO3d(mode, mode, 10, width, debug=False)
elif model_name == 'uno_3d':
    model = model_mod.Uno3D(in_channels + 3, width, pad=0, debug = False, factor=0.5)

model.to(device)

model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
model_without_ddp = model.module

print(f"Number of model parameters: {count_params(model)}")
mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
mem = (mem_params + mem_bufs) / 1024 / 1024 / 1024 # in Gb
print(f"Model uses about {mem:.2f} Gb memory")
print(f"Torch reserved {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} Gb memory for training.")

################################################################
# Training
################################################################
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5)

use_scheduler = True

if use_scheduler:

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                   step_size = 50, # Period of learning rate decay
                   gamma = 0.5)
else:
    lr_scheduler = None


training_utils.train(epochs, model, model_without_ddp, device,
                     train_loader, val_loader, optimizer, lr_scheduler, checkpoint_dir)
