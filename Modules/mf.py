import torch
from torch import nn


class MF(nn.Module):
    def __init__(self, N, M, F):
        super().__init__()
        self.N = N
        self.M = M
        self.F = F

        self.userEmbedding = nn.Embedding(M, F)
        self.itemEmbedding = nn.Embedding(N, F)

    def getUserEmbedding(self, userVectors):
        W = self.userEmbedding.weight
        return userVectors @ W
    
    def getItemEmbedding(self, itemVectors):
        W = self.itemEmbedding.weight
        return itemVectors @ W


class MFBCE(MF):
    def __init__(self, N, M, F):
        super().__init__(N, M, F)

    def forward(self, user, item):
        '''
Arguments:
    user: a M-dim tensor
    item: a N-dim tensor
Returns:
    a (batchsize,) tensor, representing the score
        '''
        userEmb = self.userEmbedding(user)
        itemEmb = self.itemEmbedding(item)

        return torch.sigmoid(torch.sum(userEmb * itemEmb, dim=1)).reshape(-1, 1)

class MFBPR(MF):
    def __init__(self, N, M, F):
        super().__init__(N, M, F)

    def forward(self, user, posItem, negItem):
        '''
Arguments:
    user: a (B, M)-dim tensor
    item: a (B, N)-dim tensor
Returns:
    a (B,) tensor, representing the score
        '''
        userEmb = self.userEmbedding(user)
        posItemEmb = self.itemEmbedding(posItem)
        negItemEmb = self.itemEmbedding(negItem)

        return -torch.sigmoid(torch.sum(userEmb * posItemEmb, dim = 1) - torch.sum(userEmb * negItemEmb, dim = 1)).reshape(-1, 1) + 2
