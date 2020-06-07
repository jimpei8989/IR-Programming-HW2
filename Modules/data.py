import os
import numpy as np
import torch

from collections import deque

from Modules.utils import *

class BCEDataset(torch.utils.data.Dataset):
    def __init__(self, mat, datadir, name='train'):
        super().__init__()
        self.mat = torch.FloatTensor(mat)

        # list of (posList, negList)
        data = pickleLoad(os.path.join(datadir, f'{name}.pkl'))
        self.X = []
        self.Y = []

        for uid, (posList, negList) in enumerate(data):
            for itemId in posList:
                self.X.append((uid, itemId))
                self.Y.append(1)
            for itemId in negList:
                self.X.append((uid, itemId))
                self.Y.append(0)

        self.X = np.stack(self.X)
        self.Y = torch.FloatTensor(np.stack(self.Y)).reshape(-1, 1)

        print(f'> Load {name} data. Size: {self.X.shape[0]}')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        uid, iid = self.X[idx]
        return self.mat[uid], self.mat[:, iid], self.Y[idx]

class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, mat, datadir, name='train', epochSize = 1000000):
        # Provided in stackoverflow
        # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        super().__init__()
        self.mat = torch.FloatTensor(mat)

        # list of (posList, negList)
        data = pickleLoad(os.path.join(datadir, f'{name}.pkl'))

        self.data = []
        for uid, (posList, negList) in enumerate(data):
            self.data += [(uid, p, n) for p in posList for n in negList]

        self.epochSize = int(epochSize) if epochSize > 1 else int(len(self.data) * epochSize)

        print(f'> Data size: {len(self.data)}')

        self.pool = deque()

    def __len__(self):
        return self.epochSize

    def __getitem__(self, index):
        if len(self.pool) == 0:
            self.pool.extend(np.random.choice(len(self.data), size = self.epochSize))

        uid, pos, neg =  self.data[self.pool.popleft()]
        return self.mat[uid], self.mat[:, pos], self.mat[:, neg]
