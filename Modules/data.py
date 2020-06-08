import os
import numpy as np
import torch

from collections import deque

from Modules.utils import *


class RandChoice:
    def __init__(self, a, cacheSize = 1000000):
        self.a = a
        self.cacheSize = cacheSize
        self.pool = deque(np.random.randint(len(self.a), size = self.cacheSize))
    
    def __call__(self):
        if len(self.pool) == 0:
            self.pool.extend(np.random.randint(len(self.a), size = self.cacheSize))
        return self.a[self.pool.popleft()]


class BCEDataset(torch.utils.data.Dataset):
    def __init__(self, mat, datadir, name='train'):
        super().__init__()
        self.mat = torch.FloatTensor(mat)

        # list of (posList, negList)
        data = pickleLoad(os.path.join(datadir, f'{name}.pkl'))
        self.posX = []
        self.negX = []

        for uid, (posList, negList) in enumerate(data):
            for itemId in posList:
                self.posX.append((uid, itemId))
            for itemId in negList:
                self.negX.append((uid, itemId))

        self.numPositive = len(self.posX)
        print(f'> Load {name} data. Positive size: {self.numPositive}')

        self.negRD = RandChoice(self.negX)

    def __len__(self):
        return self.numPositive * 2

    def __getitem__(self, idx):
        if idx < self.numPositive:
            uid, iid = self.posX[idx]
            return self.mat[uid], self.mat[:, iid], torch.FloatTensor([1])
        else:
            uid, iid = self.negRD()
            return self.mat[uid], self.mat[:, iid], torch.FloatTensor([0])


class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, mat, datadir, name='train', epochSize = 1000000):
        # Provided in stackoverflow
        # https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        super().__init__()
        self.mat = torch.FloatTensor(mat)

        # list of (posList, negList)
        data = pickleLoad(os.path.join(datadir, f'{name}.pkl'))

        num = 0
        self.uiPairs = list()
        self.negRD = dict()
        for uid, (posList, negList) in enumerate(data):
            num += len(posList) * len(negList)
            self.uiPairs.extend([(uid, p) for p in posList])
            self.negRD[uid] = RandChoice(negList, cacheSize = 5000)

        print(f'> Data size: {num}')

        self.epochSize = epochSize if epochSize > 1 else int(num * epochSize)
        self.RD = RandChoice(self.uiPairs, cacheSize=5000000)

    def __len__(self):
        return self.epochSize

    def __getitem__(self, index):
        uid, pos = self.RD()
        neg = self.negRD[uid]()
        return self.mat[uid], self.mat[:, pos], self.mat[:, neg]
