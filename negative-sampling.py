import os
import numpy as np
from argparse import ArgumentParser
from collections import deque

from sklearn.model_selection import train_test_split
from Modules.utils import *

class RandInt:
    def __init__(self, low = 0, high = 2):
        self.low = low
        self.high = high
        self.pool = deque()

    def __call__(self):
        if len(self.pool) == 0:
            self.pool.extend(np.random.randint(low = self.low, high = self.high, size = 1000000))
        return self.pool.popleft()

def main():
    args = parseArguments()

    np.random.seed(SEED)

    # Create output directory
    outputDir = os.path.join(args.datadir, args.name)
    if os.path.isdir(outputDir):
        print(f'! Warning: directory {outputDir} already exists. Exiting......')
        exit()
    else:
        os.mkdir(outputDir)
        print(f'+ Created directory {outputDir}')

    # Read data
    with EventTimer('Read Training Data'):
        with open(os.path.join(args.datadir, 'train.csv')) as f:
            data = map(lambda line : line.split(','), f.readlines()[1:])
            data = map(lambda p : list(map(int, p[1].split(' '))), data)
            data = list(data)
    
        N = len(data)
        M = max(max(d) for d in data) + 1
        print(f'> #Users: {N}')
        print(f'> #Items: {M}')

    # Interaction Matrix
    with EventTimer('Create Interaction Matrix'):
        mat = np.zeros((N, M))
        for uid, items in enumerate(data):
            mat[uid][items] = 1

        np.save(os.path.join(args.datadir, 'mat.npy'), mat)

    allData = []

    # Negative Sampling
    with EventTimer('Negative Sampling'):
        ri = RandInt(high = M)
        for uid, pos in enumerate(data):
            allItems = set(pos)
            if args.method == 'fixed':
                neg = []
                for _ in range(len(pos)):
                    while True:
                        negItem = ri()
                        if negItem not in items:
                            break
                    neg.append(negItem)
                    allItems.add(negItem)
            elif args.method == 'sample':
                neg = [item for item in range(M) if item not in pos]

            allData.append((pos, neg))

    # Split train / validation
    with EventTimer('Split Train / Validation'):
        trainData, validData = [], []
        for uid, (pos, neg) in enumerate(allData):
            p, n = int(len(pos) * 0.8), int(len(neg) * 0.8)
            np.random.shuffle(pos)
            np.random.shuffle(neg)
            trainData.append((pos[:p], neg[:n]))
            validData.append((pos[p:], neg[n:]))

    # Save to file
    with EventTimer('Save to file'):
        pickleSave(trainData, os.path.join(outputDir, 'train.pkl'))
        pickleSave(validData, os.path.join(outputDir, 'valid.pkl'))

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, help='Data directory')
    parser.add_argument('--name', type=str, help='Name for this sampling')
    parser.add_argument('--ratio', type=float, default=1.0, help='Ratio between positive & negative sampling')
    parser.add_argument('--size', type=int, default=-1, help='Number of negative samplings')
    parser.add_argument('--method', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    main()
