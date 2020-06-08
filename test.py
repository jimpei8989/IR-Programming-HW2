from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from Modules.utils import *
from Modules.mf import MF

def main():
    args = parseArguments()

    # Check if modelDir exists
    modelDir = os.path.join('Models', args.name)
    if not os.path.isdir(modelDir):
        print(f'! Warning: directory {modelDir} not found......')
        exit()

    with EventTimer('Load data / model'):
        mat = np.load(args.matrix)
        N, M = mat.shape
        print(N, M)

    with EventTimer('Inference'):
        model = MF(N, M, args.latentDim)
        model.load_state_dict(torch.load(args.weight))

        userEmbeddings = model.getUserEmbedding(mat)    # Shape (N, F)
        itemEmbeddings = model.getItemEmbedding(mat.T)  # Shape (M, F)

        predMatrix = userEmbeddings @ itemEmbeddings.T

        # To make those existing interacted item with score 0
        predictions = []
        for user, (truth, pred) in enumerate(zip(mat, predMatrix)):
            recommendations = np.argsort(pred)[::-1]
            recommendations = [item for item in recommendations if truth[item] == 0]
            predictions.append(recommendations[:50])
    
    with EventTimer('Generate prediction'):
        print(len(predictions), len(predictions[0]))
        genPredCSV(predictions, os.path.join(modelDir, 'prediction.csv'))

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--matrix')
    parser.add_argument('--weight')
    parser.add_argument('--latentDim', type=int, default=256)
    return parser.parse_args()

if __name__ == '__main__':
    main()
