#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
rbf.py: https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
"""
import torch
import torch.nn as nn
from utils import TimeDistributed

__author__ = "Yanjun Li"
__license__ = "MIT"

# RBF Layer
class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, hours_look_ahead, ref_points, in_dim, out_dim, dropout, basis_func, device):
        super(RBF, self).__init__()
        self.ref_points = ref_points
        self.device = device
        ###ref time points: ref_points
        self.interp_t = torch.linspace(0, hours_look_ahead, ref_points, device=self.device)
        self.out_dim = self.num_variables = out_dim
        self.basis_func = basis_func

        compress_fc = CompressFC(in_dim, out_dim, dropout)
        # compress_fc = CompressFCV2(in_dim, out_dim, dropout)
        self.compress_fc = TimeDistributed(compress_fc)
        self.kernel = nn.Parameter(torch.rand(out_dim, device=self.device), requires_grad=True)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     nn.init.normal_(self.centres, 0, 1)
    #     nn.init.constant_(self.sigmas, 1)

    def forward(self, interp_data, raw_input):
        """
        Each dim in c_in has corresponding beta value
        :param interp_data: with size (b, c_in, t_in). t_in = ref_points
        :param raw_input: with size (b, c_out * 4, t_out)
            [:, :features, :] observed value
            [:, features : 2*features, :] padding_mask, if no observed value, then padding zero
            [:, 2*features : 3*features, :] actual time stamps for each observation
            [:, 3*features : 4*features, :] hold out, if is 0, then this observation is taken out, used for autoencoder
        :return:
        """
        ##mask, batch_size, num_variables, all_timestamps
        m = raw_input[:, self.num_variables: 2 * self.num_variables, :]         # (b, c_out, t_out)
        ##expanded mask, each mask at each time stamp is repeated ref_points times
        mask = m[:, :, :, None].repeat(1, 1, 1, self.ref_points)      # (b, c_out, t_out, t_in)
        ##time stamps, batch_size, num_variables, all_timestamps        
        real_timestamp = raw_input[:, 2 * self.num_variables: 3 * self.num_variables, :]    # (b, c_out, t_out)
        t_out = real_timestamp.size()[2]
        centres = real_timestamp[:, :, :, None].repeat(1, 1, 1, self.ref_points)      # (b, c_out, t_out, t_in)
        distances = (centres - self.interp_t).pow(2).pow(0.5)   # make it as positive  (b, c_out, t_out, t_in)

        pos_kernel = torch.log(1 + torch.exp(self.kernel))      # make the kernel as positive
        a = torch.ones((self.out_dim, t_out, self.ref_points), device=self.device)
        beta = a * pos_kernel[:, None, None]

        # log_w_sum = torch.logsumexp(-beta * distances + torch.log(mask), dim=3)     # (b, c_out, t_out): sum over t_in
        # print(log_w_sum.size(), log_w_sum[0][0])
        # log_w_sum = log_w_sum[:, :, :, None].repeat(1, 1, 1, self.ref_points)
        # w = torch.exp(-beta * distances.pow(2) + torch.log(mask) - log_w_sum)   # divided by exp(log_w_sum), get (b, c_out, t_out, t_in)
        # print(w.size(), w[0][0])
        #
        # interp_data = interp_data.permute(0, 2, 1)      # (b, t_in, c_in)
        # interp_data = self.compress_fc(interp_data)     # (b, t_in, c_out)
        # interp_data = interp_data.permute(0, 2, 1)      # (b, c_out, t_in)
        # interp_data = interp_data[:, :, None, :].repeat(1, 1, t_out, 1)     # (b, c_out, t_out, t_in)
        # y = w * interp_data * mask     # (b, c_out, t_out, t_in)
        # y = torch.sum(y, dim=-1)

        rbf_dist = self.basis_func(beta, distances)  # (b, c_out, t_out, t_in)
        rbf_dist = rbf_dist * mask
        norm = torch.sum(rbf_dist, dim=-1)      # (b, c_out, t_out)
        # print(rbf_dist.size(), rbf_dist[0][0])
        # print('==norm: {}\n{}'.format(norm.size(), norm[0][0]))

        interp_data = interp_data.permute(0, 2, 1)  # (b, t_in, c_in)
        interp_data = self.compress_fc(interp_data)  # (b, t_in, c_out)
        interp_data = interp_data.permute(0, 2, 1)  # (b, c_out, t_in)
        interp_data = interp_data[:, :, None, :].repeat(1, 1, t_out, 1)  # (b, c_out, t_out, t_in)

        y = rbf_dist * interp_data       # (b, c_out, t_out, t_in)
        y = torch.sum(y, dim=-1) / (norm + 1e-10) * m   # (b, c_out, t_out)
        return y


class CompressFC(nn.Module):
    'The input will be (N, C), which is wrapped inside the timedistributed'
    def __init__(self, idim, odim, dropout):
        super(CompressFC, self).__init__()
        nhidden = 128
        self.model = nn.Sequential(
            nn.Linear(idim, nhidden),
            nn.BatchNorm1d(nhidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhidden, odim),
        )

    def forward(self, rec_input):
        return self.model(rec_input)

# RBFs

def gaussian(beta, alpha):
    phi = torch.exp(-beta * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
          * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
           * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases


if __name__ == '__main__':
    ref_points, hours_look_ahead = 11, 6
    b, c_out, max_t = 10, 6, 356
    num_gpus = 0
    device = torch.device("cuda" if num_gpus > 0 else "cpu")

    t_out = torch.rand((b, 4 * c_out, max_t))
    print('Out t: {}'.format(t_out.size()))

    ref_points, c_in = 20, 6
    data_in = torch.randn(b, c_in, ref_points)

    rbf = RBF(hours_look_ahead, ref_points, c_in, c_out, dropout=0.0,
              basis_func=basis_func_dict()['gaussian'], device=device)
    data_out = rbf(data_in, t_out)
    print(data_out.size())

