import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import importlib

d = np.load(
    '/p/gpfs1/kong11/data/SMART/data_assimilation_toolbox_test_data/simulation_data/normalized_training/test_data.npz')

batch_size = 1
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.Tensor(d['test_x']),
        torch.Tensor(d['test_y'])),
    batch_size=batch_size,
    shuffle=False)

in_channels = 3
width = 16
device = torch.device('cuda:3')

model_name = 'uno_3d'
model_mod = importlib.import_module(f"models.{model_name}")

if model_name == 'fno_3d':
    mode = 20
    model = model_mod.FNO3d(mode, mode, 10, width, debug=False)
elif model_name == 'uno_3d':
    model = model_mod.Uno3D(in_channels + 3, width, pad=0, debug = False, factor=0.5)

model.to(device)
checkpoint_dir = f'../output/{model_name}_8nodes_r4_g1_model2-learningRate-2/'
checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'))
model.load_state_dict(checkpoint['state_dict'])

results=[]
i = 0
for x, y in test_loader:
    i += 1
    x, y = x.to(device), y.to(device)
    out = model(x)
    results.append(out.cpu().detach().numpy())
    if i%100 == 0:
        print(i)

results = np.array(results)
np.savez_compressed(os.path.join(checkpoint_dir, 'test_results.npz'), results=results)


