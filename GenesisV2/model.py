from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from .modules import UNet, InstanceColouringSBP

from .utils import ConvGNReLU, BroadcastLayer, to_sigma, to_prior_sigma, x_loss, mask_latent_loss, get_mask_recon_stack, kl_m_loss, check_log_masks 


class GenesisV2(nn.Module):

    def __init__(self, cfg):
        super(GenesisV2, self).__init__()
        self.K_steps = cfg.model.K_steps
        self.pixel_bound = cfg.model.pixel_bound
        self.feat_dim = cfg.model.feat_dim
        self.klm_loss = cfg.model.klm_loss
        self.detach_mr_in_klm = cfg.model.detach_mr_in_klm
        self.dynamic_K = cfg.model.dynamic_K
        self.debug = cfg.general.debug
        self.encoder = UNet(
            num_blocks=int(np.log2(cfg.dataloading.img_size)-1),
            img_size=cfg.dataloading.img_size,
            filter_start=min(cfg.model.feat_dim, 64),
            in_chnls=3,
            out_chnls=cfg.model.feat_dim,
            norm='gn')
        self.encoder.final_conv = nn.Identity()
        self.att_process = InstanceColouringSBP(
            img_size=cfg.dataloading.img_size,
            kernel=cfg.model.kernel,
            colour_dim=8,
            K_steps=self.K_steps,
            feat_dim=cfg.model.feat_dim,
            semiconv=cfg.model.semiconv)
        self.seg_head = ConvGNReLU(cfg.model.feat_dim, cfg.model.feat_dim, 3, 1, 1)
        self.feat_head = nn.Sequential(
            ConvGNReLU(cfg.model.feat_dim, cfg.model.feat_dim, 3, 1, 1),
            nn.Conv2d(cfg.model.feat_dim, 2*cfg.model.feat_dim, 1))
        self.z_head = nn.Sequential(
            nn.LayerNorm(2*cfg.model.feat_dim),
            nn.Linear(2*cfg.model.feat_dim, 2*cfg.model.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*cfg.model.feat_dim, 2*cfg.model.feat_dim))
        c = cfg.model.feat_dim
        self.decoder_module = nn.Sequential(
            BroadcastLayer(cfg.dataloading.img_size // 16),
            nn.ConvTranspose2d(cfg.model.feat_dim+2, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(min(c, 64), min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.Conv2d(min(c, 64), 4, 1))
        self.autoreg_prior = cfg.model.autoreg_prior
        self.prior_lstm, self.prior_linear = None, None
        if self.autoreg_prior and self.K_steps > 1:
            self.prior_lstm = nn.LSTM(cfg.model.feat_dim, 4*cfg.model.feat_dim)
            self.prior_linear = nn.Linear(4*cfg.model.feat_dim, 2*cfg.model.feat_dim)
        assert cfg.model.pixel_std1 == cfg.model.pixel_std2
        self.std = cfg.model.pixel_std1

    def forward(self, x):
        batch_size, _, H, W = x.shape

        enc_feat, _ = self.encoder(x)
        enc_feat = F.relu(enc_feat)


        if self.dynamic_K:
            if batch_size > 1:

                log_m_k = [[] for _ in range(self.K_steps)]
                att_stats, log_s_k = None, None
                for f in torch.split(enc_feat, 1, dim=0):
                    log_m_k_b, _, _ = self.att_process(
                        self.seg_head(f), self.K_steps-1, dynamic_K=True)
                    for step in range(self.K_steps):
                        if step < len(log_m_k_b):
                            log_m_k[step].append(log_m_k_b[step])
                        else:
                            log_m_k[step].append(-1e10*torch.ones([1, 1, H, W]))
                for step in range(self.K_steps):
                    log_m_k[step] = torch.cat(log_m_k[step], dim=0)
                if self.debug:
                    assert len(log_m_k) == self.K_steps
            else:
                log_m_k, log_s_k, att_stats = self.att_process(
                    self.seg_head(enc_feat), self.K_steps-1, dynamic_K=True)
        else:
            log_m_k, log_s_k, att_stats = self.att_process(
                self.seg_head(enc_feat), self.K_steps-1, dynamic_K=False)
            if self.debug:
                assert len(log_m_k) == self.K_steps

        comp_stats = AttrDict(mu_k=[], sigma_k=[], z_k=[], kl_l_k=[], q_z_k=[])
        for log_m in log_m_k:
            mask = log_m.exp()

            obj_feat = mask * self.feat_head(enc_feat)
            obj_feat = obj_feat.sum((2, 3))

            obj_feat = obj_feat / (mask.sum((2, 3)) + 1e-5)

            mu, sigma_ps = self.z_head(obj_feat).chunk(2, dim=1)
            sigma = to_sigma(sigma_ps)
            q_z = Normal(mu, sigma)
            z = q_z.rsample()
            comp_stats['mu_k'].append(mu)
            comp_stats['sigma_k'].append(sigma)
            comp_stats['z_k'].append(z)
            comp_stats['q_z_k'].append(q_z)

        recon, x_r_k, log_m_r_k = self.decode_latents(comp_stats.z_k)


        losses = AttrDict()

        losses['err'] = x_loss(x, log_m_r_k, x_r_k, self.std)
        mx_r_k = [x*logm.exp() for x, logm in zip(x_r_k, log_m_r_k)]
        if self.klm_loss:
            if self.detach_mr_in_klm:
                log_m_r_k = [m.detach() for m in log_m_r_k]
            losses['kl_m'] = kl_m_loss(
                log_m_k=log_m_k, log_m_r_k=log_m_r_k, debug=self.debug)
        losses['kl_l_k'], p_z_k = mask_latent_loss(
            comp_stats.q_z_k, comp_stats.z_k,
            prior_lstm=self.prior_lstm, prior_linear=self.prior_linear,
            debug=self.debug)

        stats = AttrDict(
            recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k,
            log_m_r_k=log_m_r_k, mx_r_k=mx_r_k,
            instance_seg=torch.argmax(torch.cat(log_m_k, dim=1), dim=1),
            instance_seg_r=torch.argmax(torch.cat(log_m_r_k, dim=1), dim=1))

        if self.debug:
            if not self.dynamic_K:
                assert len(log_m_k) == self.K_steps
                assert len(log_m_r_k) == self.K_steps
            check_log_masks(log_m_k)
            check_log_masks(log_m_r_k)


        return recon, losses, stats, att_stats, comp_stats

    def decode_latents(self, z_k):
        x_r_k, m_r_logits_k = [], []
        for z in z_k:
            dec = self.decoder_module(z)
            x_r_k.append(dec[:, :3, :, :])
            m_r_logits_k.append(dec[:, 3: , :, :])
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        log_m_r_stack = get_mask_recon_stack(
            m_r_logits_k, 'softmax', log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        recon = (m_r_stack * x_r_stack).sum(dim=4)

        return recon, x_r_k, log_m_r_k

    def sample(self, batch_size, K_steps=None):
        K_steps = self.K_steps if K_steps is None else K_steps

        if self.autoreg_prior:
            z_k = [Normal(0, 1).sample([batch_size, self.feat_dim])]
            state = None
            for k in range(1, K_steps):
                lstm_out, state = self.prior_lstm(
                    z_k[-1].view(1, batch_size, -1), state)
                linear_out = self.prior_linear(lstm_out)
                linear_out = torch.chunk(linear_out, 2, dim=2)
                linear_out = [item.squeeze(0) for item in linear_out]
                mu = torch.tanh(linear_out[0])
                sigma = to_prior_sigma(linear_out[1])
                p_z = Normal(mu.view([batch_size, self.feat_dim]),
                             sigma.view([batch_size, self.feat_dim]))
                z_k.append(p_z.sample())
        else:
            p_z = Normal(0, 1)
            z_k = [p_z.sample([batch_size, self.feat_dim])
                   for _ in range(K_steps)]

        recon, x_r_k, log_m_r_k = self.decode_latents(z_k)

        stats = AttrDict(x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x*m.exp() for x, m in zip(x_r_k, log_m_r_k)])
        return recon, stats