import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict

import numpy as np

from .utils import ConvGNReLU, ConvINReLU, ConvReLU, Flatten, SemiConv, euclidian_distance, squared_distance, clamp_preserve_gradients

class UNet(nn.Module):

    def __init__(self, num_blocks, img_size=64,
                 filter_start=32, in_chnls=4, out_chnls=1,
                 norm='in'):
        super(UNet, self).__init__()
        c = filter_start
        if norm == 'in':
            conv_block = ConvINReLU
        elif norm == 'gn':
            conv_block = ConvGNReLU
        else:
            conv_block = ConvReLU
        if num_blocks == 4:
            enc_in = [in_chnls, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [in_chnls, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]
        self.down = []
        self.up = []
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.featuremap_size = img_size // 2**(num_blocks-1)
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(2*c*self.featuremap_size**2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2*c*self.featuremap_size**2), nn.ReLU()
        )
        self.final_conv = nn.Conv2d(c, out_chnls, 1)
        self.out_chnls = out_chnls

    def forward(self, x):
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down)-1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            x_down.append(act)
        
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1,
                         self.featuremap_size, self.featuremap_size)
        
        for i, block in enumerate(self.up):
            if x_up.size(2) != skip[-1 - i].size(2):
                diff = skip[-1 - i].size(2) - x_up.size(2)
                x_up = F.pad(x_up, (diff // 2, diff - diff // 2, diff // 2, diff - diff // 2))
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up)-1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        return self.final_conv(x_up), None



class InstanceColouringSBP(nn.Module):

    def __init__(self, img_size, kernel='gaussian',
                 colour_dim=8, K_steps=None, feat_dim=None,
                 semiconv=True):
        super(InstanceColouringSBP, self).__init__()

        self.img_size = img_size
        self.kernel = kernel
        self.colour_dim = colour_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.kernel == 'laplacian':
            sigma_init = 1.0 / (np.sqrt(K_steps)*np.log(2))
        elif self.kernel == 'gaussian':
            sigma_init = 1.0 / (K_steps*np.log(2))
        elif self.kernel == 'epanechnikov':
            sigma_init = 2.0 / K_steps
        else:
            return ValueError("No valid kernel.")
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init, device=self.device).log())

        if semiconv:
            self.colour_head = SemiConv(feat_dim, self.colour_dim, img_size)
        else:
            self.colour_head = nn.Conv2d(feat_dim, self.colour_dim, 1)

    def forward(self, features, steps_to_run, debug=False,
                dynamic_K=False, *args, **kwargs):
        batch_size = features.size(0)
        stats = AttrDict()
        if isinstance(features, tuple):
            features = features[0]
        if dynamic_K:
            assert batch_size == 1

        colour_out = self.colour_head(features)
        if isinstance(colour_out, tuple):
            colour, delta = colour_out
        else:
            colour, delta = colour_out, None

        rand_pixel = torch.empty(batch_size, 1, *colour.shape[2:], device=self.device)
        rand_pixel = rand_pixel.uniform_()

        seed_list = []
        log_m_k = []
        log_s_k = [torch.zeros(batch_size, 1, self.img_size, self.img_size, device=self.device)]
        for step in range(steps_to_run):

            scope = F.interpolate(log_s_k[step].exp(), size=colour.shape[2:],
                                  mode='bilinear', align_corners=False)
            pixel_probs = rand_pixel * scope
            rand_max = pixel_probs.flatten(2).argmax(2).flatten()

            seed = torch.empty((batch_size, self.colour_dim), device=self.device)
            for bidx in range(batch_size):
                seed[bidx, :] = colour.flatten(2)[bidx, :, rand_max[bidx]]
            seed_list.append(seed)

            if self.kernel == 'laplacian':
                distance = euclidian_distance(colour, seed)
                alpha = torch.exp(- distance / self.log_sigma.exp())
            elif self.kernel == 'gaussian':
                distance = squared_distance(colour, seed)
                alpha = torch.exp(- distance / self.log_sigma.exp())
            elif self.kernel == 'epanechnikov':
                distance = squared_distance(colour, seed)
                alpha = (1 - distance / self.log_sigma.exp()).relu()
            else:
                raise ValueError("No valid kernel.")
            alpha = alpha.unsqueeze(1)

            if debug:
                assert alpha.max() <= 1, alpha.max()
                assert alpha.min() >= 0, alpha.min()

            alpha = clamp_preserve_gradients(alpha, 0.01, 0.99)

            log_a = torch.log(alpha)
            log_neg_a = torch.log(1 - alpha)
            log_m = log_s_k[step] + log_a
            if dynamic_K and log_m.exp().sum() < 20:
                break
            log_m_k.append(log_m)
            log_s_k.append(log_s_k[step] + log_neg_a)

        log_m_k.append(log_s_k[-1])

        stats.update({'colour': colour, 'delta': delta, 'seeds': seed_list})
        return log_m_k, log_s_k, stats