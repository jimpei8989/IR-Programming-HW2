import os
import numpy as np
import torch

from Modules import utils

class RSDataset(torch.utils.data.Dataset):
    def __init__(self, mat, datadir, name='train'):
        super().__init__()
        self.mat = torch.FloatTensor(mat)
        self.X = np.load(os.path.join(datadir, f'{name}X.npy'))
        self.Y = torch.FloatTensor(np.load(os.path.join(datadir, f'{name}Y.npy'))).reshape(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        uid, iid = self.X[idx]
        return self.mat[uid], self.mat[:, iid], self.Y[idx]
