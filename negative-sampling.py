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

    # Negative Sampling
    with EventTimer('Negative Sampling'):
        X, Y = [], []
        ri = RandInt(high = M)
        for uid, items in enumerate(data):
            X += [(uid, item) for item in items]
            Y += [1] * len(items)

            m = len(items)
            items = set(items)
            for _ in range(m):
                while True:
                    negItem = ri()
                    if negItem not in items:
                        break
                X.append((uid, negItem))
                items.add(negItem)
                Y.append(0)

        X = np.stack(list(map(lambda p : np.array(p), X)))
        Y = np.stack(Y)

        print(f'> X.shape: {X.shape}')
        print(f'> Y.shape: {Y.shape}')

    # Split
    with EventTimer('Split data'):
        trainX, validX, trainY, validY = train_test_split(X, Y, test_size = 0.2, random_state = SEED)
        print(f'> training\tX: {trainX.shape}\t| Y: {trainY.shape}')
        print(f'> validation\tX: {validX.shape}\t| Y: {validY.shape}')
    
    # Write to file
    with EventTimer('Write to file'):
        np.save(os.path.join(outputDir, 'X.npy'), X)
        np.save(os.path.join(outputDir, 'Y.npy'), Y)
        np.save(os.path.join(outputDir, 'trainX.npy'), trainX)
        np.save(os.path.join(outputDir, 'trainY.npy'), trainY)
        np.save(os.path.join(outputDir, 'validX.npy'), validX)
        np.save(os.path.join(outputDir, 'validY.npy'), validY)

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
