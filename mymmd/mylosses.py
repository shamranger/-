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


    # # def _init_targets(self):
    # #     c1_y = self.target.mean(dim=self.axis).view(1, -1, 1, 1)
    # #     m_ys = list()
    # #     for i in range(2, self.k + 1):
    # #         # watch out: zeroth element is pow 2, first is pow 3...
    # #         m_ys.append((self.target - c1_y).pow(i).mean(dim=self.axis))
    # #     return c1_y, m_ys
    #
    #
    def __call__(self, x, y):
        #         l = prediction - target
        #         l = l**2
        #         return torch.sqrt(torch.mean(l))
        #print(x.shape)
        #print(x.transpose(2,3).shape)
        #print(y.shape)
        n1 = x.shape[3]
        n2 = y.shape[3]
        #idx, idy = torch.argmax(x), torch.argmax(y)
        #print(idx, idy, x[:, idx], y[:, idy])
        diffx = torch.matmul(x, x.transpose(2,3)).sum(axis=[0,2,3],keepdim=False)
        diffy = torch.matmul(y, y.transpose(2,3)).sum(axis=[0,2,3],keepdim=False)
        diffxy = torch.matmul(x, y.transpose(2,3)).sum(axis=[0,2,3],keepdim=False)
        #print(diffx.shape,diffy.shape,diffxy.shape)
        diffs = diffx + diffy - 2 * diffxy
        diff = diffs.mean()/(n1*n2)
        return diff

    # def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
    #     super(StyleLoss, self).__init__()
    #     self.kernel_num = kernel_num
    #     self.kernel_mul = kernel_mul
    #     self.fix_sigma = None
    #     self.kernel_type = kernel_type
    #
    # def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
    #     source = source.squeeze()
    #     target = target.squeeze()
    #     n_samples = int(source.size()[0]) + int(target.size()[0])
    #     total = torch.cat([source, target], dim=0)
    #     total0 = total.unsqueeze(0).expand(
    #         int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))).view(int(total.size(0)), int(total.size(0)),-1)
    #     total1 = total.unsqueeze(1).expand(
    #         int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2))).view(int(total.size(0)), int(total.size(0)),-1)
    #     print(total1.shape,total0.shape)
    #     total0 = total0-total1
    #     L2_distance = ((total0-total1)**2).sum(axis=[2, 3], keepdim=False)
    #     if fix_sigma:
    #         bandwidth = fix_sigma
    #     else:
    #         bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #     bandwidth /= kernel_mul ** (kernel_num // 2)
    #     bandwidth_list = [bandwidth * (kernel_mul**i)
    #                       for i in range(kernel_num)]
    #     kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
    #                   for bandwidth_temp in bandwidth_list]
    #     return sum(kernel_val)
    #
    # def linear_mmd2(self, f_of_X, f_of_Y):
    #     loss = 0.0
    #     delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
    #     loss = delta.dot(delta.T)
    #     return loss
    #
    # def forward(self, source, target):
    #     if self.kernel_type == 'linear':
    #         return self.linear_mmd2(source, target)
    #     elif self.kernel_type == 'rbf':
    #         batch_size = int(source.size()[0])
    #         kernels = self.guassian_kernel(
    #             source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
    #         XX = torch.mean(kernels[:batch_size, :batch_size])
    #         YY = torch.mean(kernels[batch_size:, batch_size:])
    #         XY = torch.mean(kernels[:batch_size, batch_size:])
    #         YX = torch.mean(kernels[batch_size:, :batch_size])
    #         loss = torch.mean(XX + YY - XY - YX)
    #         return loss
    #


# import torch
# import torch.nn as nn
#
# class MMDLoss(nn.Module):
#     '''
#     计算源域数据和目标域数据的MMD距离
#     Params:
#     source: 源域数据（n * len(x))
#     target: 目标域数据（m * len(y))
#     kernel_mul:
#     kernel_num: 取不同高斯核的数量
#     fix_sigma: 不同高斯核的sigma值
#     Return:
#     loss: MMD loss
#     '''
#     def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
#         super(MMDLoss, self).__init__()
#         self.kernel_num = kernel_num
#         self.kernel_mul = kernel_mul
#         self.fix_sigma = None
#         self.kernel_type = kernel_type
#
#     def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
#         n_samples = int(source.size()[0]) + int(target.size()[0])
#         total = torch.cat([source, target], dim=0)
#         total0 = total.unsqueeze(0).expand(
#             int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         total1 = total.unsqueeze(1).expand(
#             int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         L2_distance = ((total0-total1)**2).sum(2)
#         if fix_sigma:
#             bandwidth = fix_sigma
#         else:
#             bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
#         bandwidth /= kernel_mul ** (kernel_num // 2)
#         bandwidth_list = [bandwidth * (kernel_mul**i)
#                           for i in range(kernel_num)]
#         kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
#                       for bandwidth_temp in bandwidth_list]
#         return sum(kernel_val)
#
#     def linear_mmd2(self, f_of_X, f_of_Y):
#         loss = 0.0
#         delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
#         loss = delta.dot(delta.T)
#         return loss
#
#     def forward(self, source, target):
#         if self.kernel_type == 'linear':
#             return self.linear_mmd2(source, target)
#         elif self.kernel_type == 'rbf':
#             batch_size = int(source.size()[0])
#             kernels = self.guassian_kernel(
#                 source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#             XX = torch.mean(kernels[:batch_size, :batch_size])
#             YY = torch.mean(kernels[batch_size:, batch_size:])
#             XY = torch.mean(kernels[:batch_size, batch_size:])
#             YX = torch.mean(kernels[batch_size:, :batch_size])
#             loss = torch.mean(XX + YY - XY - YX)
#             return loss
