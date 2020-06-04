import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, K, F):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(K, F), requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.W)

class MF(nn.Module):
    def __init__(self, N, M, F):
        super().__init__()
        self.N = N
        self.M = M
        self.F = F

        self.userEmbedding = Embedding(M, F)
        self.itemEmbedding = Embedding(N, F)

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

        return torch.sum(userEmb * itemEmb, dim = 1)