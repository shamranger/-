import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RMSELoss(nn.Module):
    def __init__(self, eps=0):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()

    def __call__(self, prediction, target):
        #         l = prediction - target
        #         l = l**2
        #         return torch.sqrt(torch.mean(l))
        return F.mse_loss(prediction, target)


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
    def __call__(self, x, y):
        #         l = prediction - target
        #         l = l**2
        #         return torch.sqrt(torch.mean(l))
        #print(x.shape)
        #print(x.transpose(2,3).shape)
        #print(y.shape)
        n1 = x.shape[3]
        # n2 = y.shape[3]
        #idx, idy = torch.argmax(x), torch.argmax(y)
        #print(idx, idy, x[:, idx], y[:, idy])
        diffx = torch.matmul(x, x.transpose(2,3)).pow(2).mean(axis=[0,2,3],keepdim=False)
        diffy = torch.matmul(y, y.transpose(2,3)).pow(2).mean(axis=[0,2,3],keepdim=False)
        diffxy = torch.matmul(x, y.transpose(2,3)).pow(2).mean(axis=[0,2,3],keepdim=False)
        #print(diffx.shape,diffy.shape,diffxy.shape)
        diffs = diffx + diffy - 2 * diffxy
        # diff = diffs.mean()/(n1*n2)
        diff = diffs.mean()/n1*1e-1*10
        return diff
