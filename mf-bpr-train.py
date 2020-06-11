import json
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from Modules.utils import *
from Modules.data import BPRDataset
from Modules.mf import MFBPR

class BPRLoss(nn.Module):
    def  __init__(self):
        super().__init__()
    
    def forward(self, output):
        return torch.mean(torch.log(output))

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
                'l2': args.l2,
                'batch': args.batch,
            }), file = f)
        
    with EventTimer('Load data'):
        mat = np.load(args.matrix)

        # Should be 4454 * 3260
        N, M = mat.shape

        trainDataset = BPRDataset(mat, args.datadir, 'train', args.epochSize)
        validDataset = BPRDataset(mat, args.datadir, 'valid', 256000)
        trainDataloader = DataLoader(trainDataset, batch_size = args.batch, shuffle = True, num_workers=8)
        validDataloader = DataLoader(validDataset, batch_size = args.batch, shuffle = False, num_workers=8)

    with EventTimer('Load Model'):
        model = MFBPR(N, M, args.latentDim).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.l2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=5e-2, patience=5, min_lr=1e-7, verbose=True)

        if args.since > 0:
            checkpoint = torch.load(os.path.join(modelDir, f'checkpoint-{args.since:03d}.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    with EventTimer('Train Model'):
        criterion = BPRLoss()

        def run_epoch(dataloader, train = True):
            losses = []
            with (torch.enable_grad() if train else torch.no_grad()):
                if train: model.train()
                else: model.eval()

                for userVec, posItemVec, negItemVec in tqdm(dataloader):
                    if train: optimizer.zero_grad()

                    output = model(userVec.cuda(), posItemVec.cuda(), negItemVec.cuda()).cpu()

                    loss = criterion(output)

                    if train:
                        loss.backward()
                        optimizer.step()

                    losses.append(loss.item())

            return np.mean(losses)

        for epoch in range(1, args.epochs + 1):
            with EventTimer(verbose=False) as et:
                trainLoss = run_epoch(trainDataloader, True)
                validLoss = run_epoch(validDataloader, False)

                scheduler.step(validLoss)
                print(f'> Epoch {epoch:03d} / {args.epochs}: [{et.gettime():.4f}s] Train Loss: {trainLoss:.6f} | Valid Loss: {validLoss:.6f}')

            if epoch % 10 == 0:
                model.cpu()
                torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                    }, os.path.join(modelDir, f'checkpoint-{epoch:03d}.pth'))
                model.cuda()

    with EventTimer('Save model'):
        model.cpu()
        torch.save(model.cpu().state_dict(), os.path.join(modelDir, 'final-weight.pth'))

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--datadir')
    parser.add_argument('--matrix')
    parser.add_argument('--latentDim', type=int, default=256)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--epochSize', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--since', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    main()
