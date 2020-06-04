from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from Modules.utils import *
from Modules.data import RSDataset
from Modules.mf import MF

def main():
    args = parseArguments()

    with EventTimer('Load data'):
        mat = np.load(args.matrix)

        # Should be 4455 * 3260
        N, M = mat.shape

        trainDataset = RSDataset(mat, args.datadir, 'train')
        validDataset = RSDataset(mat, args.datadir, 'valid')
        trainDataloader = DataLoader(trainDataset, batch_size = args.batch)
        validDataloader = DataLoader(validDataset, batch_size = args.batch)

    with EventTimer('Train Model'):
        model = MF(N, M, args.latentDim)

        optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
        criterion = nn.BCEWithLogitsLoss()

        def run_epoch(dataloader, train = True):
            losses = []
            with (torch.enable_grad() if train else torch.no_grad()):
                if train: model.train()
                else: model.eval()

                for userVec, itemVec, y in tqdm(dataloader):
                    if train: optimizer.zero_grad()

                    output = model(userVec, itemVec)

                    loss = criterion(output, y)

                    if train:
                        loss.backward()
                        optimizer.step()

                    scores = torch.sigmoid(output.detach()).numpy()
                    accu = np.mean((scores > 0.5) == y.numpy())

                    losses.append((loss.item(), accu))

            return map(np.mean, zip(*losses))

        for epoch in range(1, args.epochs + 1):
            trainLoss, trainAccu = run_epoch(trainDataloader, True)
            validLoss, validAccu = run_epoch(validDataloader, False)

            print(f'> Epoch {epoch} / {args.epochs}: Train BCE: {trainLoss:.6f} ; Accu: {trainAccu} | Valid BCE: {validLoss:.6f} ; Accu: {validAccu}')

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--datadir')
    parser.add_argument('--matrix')
    parser.add_argument('--latentDim', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()

if __name__ == '__main__':
    main()
