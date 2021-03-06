import json
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from Modules.utils import *
from Modules.data import BCEDatasetSample, BCEDatasetFixed
from Modules.mf import MFBCE

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

        if args.dataset == 'sample':
            BCEDataset = BCEDatasetSample
        elif args.dataset == 'fixed':
            BCEDataset = BCEDatasetFixed

        trainDataset = BCEDataset(mat, args.datadir, 'train')
        validDataset = BCEDataset(mat, args.datadir, 'valid')
        trainDataloader = DataLoader(trainDataset, batch_size = args.batch, shuffle = True, num_workers=4)
        validDataloader = DataLoader(validDataset, batch_size = args.batch, shuffle = False, num_workers=4)

    with EventTimer('Train Model'):
        model = MFBCE(N, M, args.latentDim).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=5e-2, patience=5, min_lr=1e-7, verbose=True)

        criterion = nn.BCELoss()

        def run_epoch(dataloader, train = True):
            losses = []
            with (torch.enable_grad() if train else torch.no_grad()):
                if train: model.train()
                else: model.eval()

                for userVec, itemVec, y in tqdm(dataloader, ncols=80):
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
            print(f'> Epoch {epoch:3d}/{args.epochs}: [{et.gettime():.4f}s] Train Loss: {trainLoss:.6f} ; Accu: {trainAccu:.4f} | Valid Loss: {validLoss:.6f} ; Accu: {validAccu:.4f}')

            if epoch % 10 == 0:
                torch.save(model.cpu().state_dict(), os.path.join(modelDir, f'checkpoint-{epoch:03d}.pth'))
                model.cuda()

    with EventTimer('Save model'):
        model.cpu()
        torch.save(model.state_dict(), os.path.join(modelDir, 'final-weight.pth'))

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--datadir')
    parser.add_argument('--matrix')
    parser.add_argument('--dataset', type=str, default='sample')
    parser.add_argument('--latentDim', type=int, default=256)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--l2', type=float, default=1e-4)
    return parser.parse_args()

if __name__ == '__main__':
    main()
