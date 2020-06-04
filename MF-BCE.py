import json
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

    # Check if modelDir exists
    with EventTimer('Checking configs'):
        modelDir = os.path.join('Models', args.name)
        if os.path.isdir(modelDir):
            print(f'! Warning: directory {modelDir} already exists.')
            # exit()
        else:
            os.mkdir(modelDir)
            print(f'+ Created directory {modelDir}')

        with open(os.path.join(modelDir, 'hyperparameters.json'), 'w') as f:
            print(json.dumps({
                'latentDims': args.latentDim,
                'epochs': args.epochs,
                'lr': args.lr,
                'batch': args.batch,
            }), file = f)
        
    with EventTimer('Load data'):
        mat = np.load(args.matrix)

        # Should be 4454 * 3260
        N, M = mat.shape

        trainDataset = RSDataset(mat, args.datadir, 'train')
        validDataset = RSDataset(mat, args.datadir, 'valid')
        trainDataloader = DataLoader(trainDataset, batch_size = args.batch)
        validDataloader = DataLoader(validDataset, batch_size = args.batch)

    with EventTimer('Train Model'):
        model = MF(N, M, args.latentDim).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=0.5, patience=10, min_lr=1e-7, verbose=True)
        criterion = nn.BCELoss()

        def run_epoch(dataloader, train = True):
            losses = []
            with (torch.enable_grad() if train else torch.no_grad()):
                if train: model.train()
                else: model.eval()

                for userVec, itemVec, y in tqdm(dataloader):
                    if train: optimizer.zero_grad()

                    output = model(userVec.cuda(), itemVec.cuda()).cpu()

                    loss = criterion(output, y.reshape(-1, 1))

                    if train:
                        loss.backward()
                        optimizer.step()

                    scores = torch.sigmoid(output.detach()).numpy()
                    accu = np.mean((scores > 0.5) == y.numpy())

                    losses.append((loss.item(), accu))

            return map(np.mean, zip(*losses))

        for epoch in range(1, args.epochs + 1):
            with EventTimer(verbose=False) as et:
                trainLoss, trainAccu = run_epoch(trainDataloader, True)
                validLoss, validAccu = run_epoch(validDataloader, False)

                scheduler.step(validLoss)
                print(f'> Epoch {epoch} / {args.epochs}: [{et.gettime():.4f}s] Train BCE: {trainLoss:.6f} ; Accu: {trainAccu:.4f} | Valid BCE: {validLoss:.6f} ; Accu: {validAccu:.4f}')

    with EventTimer('Save model'):
        model.cpu()
        torch.save(model.state_dict(), os.path.join(modelDir, 'final-weight.pth'))

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--datadir')
    parser.add_argument('--matrix')
    parser.add_argument('--latentDim', type=int, default=256)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--l2', type=float, default=1e-6)
    return parser.parse_args()

if __name__ == '__main__':
    main()
