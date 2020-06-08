import torch
from torch import nn


class Factorization(nn.Module):
    def __init__(self, K, F):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(K, F), requires_grad=True)
    
    def forward(self, x):
        return torch.matmul(x, self.weight)


class MF(nn.Module):
    def __init__(self, N, M, F):
        super().__init__()
        self.N = N
        self.M = M
        self.F = F

        self.userEmbedding = Factorization(M, F) 
        self.itemEmbedding = Factorization(N, F)

    def getUserEmbedding(self, userVectors):
        return userVectors @ self.userEmbedding.weight.detach().numpy()
    
    def getItemEmbedding(self, itemVectors):
        return itemVectors @ self.itemEmbedding.weight.detach().numpy()


class MFBCE(MF):
    def __init__(self, N, M, F):
        super().__init__(N, M, F)

    def forward(self, userVec, itemVec):
        '''
Arguments:
    user: a M-dim tensor
    item: a N-dim tensor
Returns:
    a (batchsize,) tensor, representing the score
        '''
        userEmb = self.userEmbedding(userVec)
        itemEmb = self.itemEmbedding(itemVec)

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
