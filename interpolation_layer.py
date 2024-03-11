#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
interpolation_layer_pyt.py: 
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SingleChannelInterp(nn.Module):
    ###ref_points, hours_from_admission, num_variables, num_timestamps, device
    def __init__(self, ref_points, hours_look_ahead, d_dim, timestamp, device, activation="sigmoid"):
        super(SingleChannelInterp, self).__init__()
        self.ref_points = ref_points
        self.hours_look_ahead = hours_look_ahead  # in hours
        self.activation = activation
        self.device = device

        self.timestamp = timestamp
        self.d_dim = d_dim
        self.kernel = nn.Parameter(torch.rand(self.d_dim, device=self.device), requires_grad=True)
        # Each variable has different kernel.

    # input_shape [batch, features * 4, time_stamp]
    # [:, :features, :] observed value
    # [:, features : 2*features, :] padding_mask, if no observed value, then padding zero
    # [:, 2*features : 3*features, :] actual time stamps for each observation
    # [:, 3*features : 4*features, :] hold out, if is 0, then this observation is taken out, used for autoencoder
    def forward(self, x):
        x_t = x[:, :self.d_dim, :]
        d = x[:, 2 * self.d_dim:3 * self.d_dim, :]

        # if reconstruction:
        #     output_dim = self.timestamp
        #     m = x[:, 3 * self.d_dim:, :]    # ae_mask
        #     ref_t = d[:, :, None, :].repeat(1, 1, output_dim, 1)
        # else:
        m = x[:, self.d_dim: 2 * self.d_dim, :]     # padding_mask
        ref_t = torch.linspace(0, self.hours_look_ahead, self.ref_points, device=self.device)
        output_dim = self.ref_points
        # ref_t.shape = (1, ref_t.shape[0])

        # x_t = x_t*m
        d = d[:, :, :, None].repeat(1, 1, 1, output_dim)
        mask = m[:, :, :, None].repeat(1, 1, 1, output_dim)
        x_t = x_t[:, :, :, None].repeat(1, 1, 1, output_dim)
        norm = (d - ref_t) * (d - ref_t)
        a = torch.ones((self.d_dim, self.timestamp, output_dim), device=self.device)
        pos_kernel = torch.log(1 + torch.exp(self.kernel))
        alpha = a * pos_kernel[:, None, None]

        # Intensity channel
        # t1 = -alpha * norm
        # t2 = torch.log(mask)
        # t3 = t1 + t2
        # t4 = torch.exp(t3)
        w = torch.logsumexp(-alpha * norm + torch.log(mask), dim=2)     # (b, c, r)

        # low-pass channel
        w1 = w[:, :, None, :].repeat(1, 1, self.timestamp, 1)
        w1 = torch.exp(-alpha * norm + torch.log(mask) - w1)
        y = torch.sum(w1 * x_t, dim=2)

        """Original Version: reconstruction is different from prediction
        if reconstruction:
            rep1 = torch.cat([y, w], 1)    
        else:
            # high-pass channel
            w_t = torch.logsumexp(-10.0 * alpha * norm + torch.log(mask), dim=2)  # kappa = 10
            w_t = w_t[:, :, None, :].repeat(1, 1, self.timestamp, 1)
            w_t = torch.exp(-10.0 * alpha * norm + torch.log(mask) - w_t)
            y_trans = torch.sum(w_t * x_t, dim=2)
            rep1 = torch.cat([y, w, y_trans], 1)    # output: (b, 3*c, r)
        rep1 = rep1.permute(0, 2, 1)    # output: (b, t, 3*c)
        """

        # high-pass channel
        w_t = torch.logsumexp(-10.0 * alpha * norm + torch.log(mask), dim=2)  # kappa = 10
        w_t = w_t[:, :, None, :].repeat(1, 1, self.timestamp, 1)
        w_t = torch.exp(-10.0 * alpha * norm + torch.log(mask) - w_t)
        y_trans = torch.sum(w_t * x_t, dim=2)
        rep1 = torch.cat([y, w, y_trans], 1)  # output: (b, 3*c, r)
        rep1 = rep1.permute(0, 2, 1)
        return rep1


class CrossChannelInterp(nn.Module):
    def __init__(self, d_dim, timestamp, device, activation="sigmoid"):
        super(CrossChannelInterp, self).__init__()
        self.d_dim = d_dim
        self.timestamp = timestamp
        self.device = device
        self.activation = activation
        gain = 1.0
        self.kernel = nn.Parameter(gain * torch.eye(self.d_dim, self.d_dim, device=self.device), requires_grad=True)

    def forward(self, x, reconstruction=False):
        x = x.permute(0, 2, 1)      # (b, 3*c, t)
        self.output_dim = x.size()[-1]      # output_dim = time_step
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim: 2 * self.d_dim, :]
        intensity = torch.exp(w)
        y = y.permute(0, 2, 1)
        w = w.permute(0, 2, 1)
        w2 = w
        w = w[:, :, :, None].repeat(1, 1, 1, self.d_dim)
        den = torch.logsumexp(w, dim=2)
        w = torch.exp(w2 - den)
        mean = torch.mean(y, dim=1)
        mean = mean[:, None, :].repeat(1, self.output_dim, 1)
        w2 = torch.matmul(w * (y - mean), self.kernel) + mean   # (B, T, C)
        rep1 = w2.permute(0, 2, 1)  # (B, C, T)

        """
        if reconstruction is False:
            y_trans = x[:, 2 * self.d_dim:3 * self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = torch.cat([rep1, intensity, y_trans], 1)     # (B, C, T)
        """
        y_trans = x[:, 2 * self.d_dim:3 * self.d_dim, :]
        y_trans = y_trans - rep1  # subtracting smooth from transient part
        rep1 = torch.cat([rep1, intensity, y_trans], 1)  # (B, C, T)

        rep1 = rep1.permute(0, 2, 1)    # (B, T, C)
        return rep1


if __name__ == '__main__':
    ref_points, hours_look_ahead = 11, 6
    b, c, max_t = 10, 6, 30
    num_gpus = 0
    device = torch.device("cuda" if num_gpus > 0 else "cpu")

    sci = SingleChannelInterp(ref_points, hours_look_ahead, c, max_t, device)
    cci = CrossChannelInterp(c, max_t, device)

    feat = torch.randn(b, c, max_t)
    print('Input feat: {}'.format(feat.size()))
    mask = torch.randint(0, 2, size=(b, c, max_t), dtype=torch.float32)
    timestamp = ref_points * torch.rand((b, c, max_t))
    hold_out = torch.randint(0, 2, size=(b, c, max_t), dtype=torch.float32)

    x = torch.cat([feat, mask, timestamp, hold_out], dim=1)

    x1 = sci(x)
    print('Feat extracted (x1) shape: {}'.format(x1.size()))
    interp_x = cci(x1)
    print('Feat extracted (interp_x) shape: {}'.format(interp_x.size()))
