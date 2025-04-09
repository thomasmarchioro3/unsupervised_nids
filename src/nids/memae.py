import math
import torch

import torch.nn as nn
import torch.nn.functional as F


# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim: int, latent_dim: int, shrink_thres: float=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.latent_dim = latent_dim
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.latent_dim))  # M x C
        self.bias = None
        self.shrink_thres= shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres>0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        output = F.linear(att_weight, mem_trans)  # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


class MemAE(nn.Module):
    def __init__(self, num_features: int, mem_dim: int, shrink_thres: float=0.0025):
        super(MemAE, self).__init__()

        self.num_features = num_features
        self.mem_dim = mem_dim
        self.shrink_thres = shrink_thres

        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, 60),
            nn.Tanh(),
            nn.Linear(60, 30),
            nn.Tanh(),
            nn.Linear(30, 10),
            nn.Tanh(),
            nn.Linear(10, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.Tanh(),
            nn.Linear(10, 30),
            nn.Tanh(),
            nn.Linear(30, 60),
            nn.Tanh(),
            nn.Linear(60, self.num_features),
        )

        self.memory = MemoryUnit(mem_dim, 3, shrink_thres=self.shrink_thres)

    def forward(self, x, return_att=True):
        encoded = self.encoder(x)
        out_mem = self.memory(encoded)
        decoded = self.decoder(out_mem['output'])

        if not return_att:
            return decoded
        return {'output': decoded, 'att': out_mem['att']}
    
    def evaluate_errors(self, x, metric='rmse'):

        with torch.no_grad():
            x_recon = self.forward(x)['output']

            if metric == 'rmse':
                errors = torch.sqrt(torch.mean((x_recon - x) ** 2, dim=1))  # RMSE per sample
            elif metric == 'mae':
                errors = torch.mean(torch.abs(x_recon['output'] - x), dim=1)  # MAE per sample
            else:
                raise NotImplementedError

        return errors
    
class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b