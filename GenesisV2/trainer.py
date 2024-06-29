import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from .utils import GECO
from utils.misc import average_ari, colour_seg_masks


class Trainer:

    def __init__(self, model, optimizer, device, config):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.step = 0

        self.model.to(self.device)

        if config.loss.geco:
            num_elements = 3 * config.dataloading.img_size**2 
            geco_goal = config.loss.g_goal * num_elements
            geco_lr = float(config.loss.g_lr) * (64**2 / config.dataloading.img_size**2)
            self.geco = GECO(geco_goal, geco_lr, config.loss.g_alpha, float(config.loss.g_init), float(config.loss.g_min), float(config.loss.g_speedup))
            self.beta = self.geco.beta
            self.geco.to_cuda()
        else:
            self.beta = torch.tensor(config.loss.beta)

    def train_step(self, inputs):
        self.model.train()
        inputs = inputs.to(self.device)
        self.optimizer.zero_grad()
        output, losses, stats, att_stats, comp_stats = self.model(inputs)
        
        err = losses.err.mean(0)
        kl_m, kl_l = torch.tensor(0), torch.tensor(0)
        if 'kl_m' in losses:
            kl_m = losses.kl_m.mean(0)
        elif 'kl_m_k' in losses:
            kl_m = torch.stack(losses.kl_m_k, dim=1).mean(dim=0).sum()
        if 'kl_l' in losses:
            kl_l = losses.kl_l.mean(0)
        elif 'kl_l_k' in losses:
            kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()

        elbo = (err + kl_l+ kl_m).detach()

        if self.config.loss.geco:
            self.beta = self.geco.beta
            loss = self.geco.loss(err, kl_l + kl_m)
        else:
            if self.config.loss.beta_warmup:
                self.beta = self.config.loss.beta * self.step / (0.2 * self.config.training.epochs)
                self.beta = torch.tensor(self.beta).clamp(0, self.config.loss.beta)
            else:
                self.beta = self.config.loss.beta
            loss = err + self.beta * (kl_l + kl_m)

        loss.backward()
        self.optimizer.step()

        return err, loss.item(), elbo

    def test_step(self, inputs):
        self.model.eval()
        inputs = inputs.to(self.device)
        with torch.no_grad():
            output, losses, _, _, _ = self.model(inputs)
            err = losses.err.mean(0)
            kl_m, kl_l = torch.tensor(0), torch.tensor(0)
            if 'kl_m' in losses:
                kl_m = losses.kl_m.mean(0)
            elif 'kl_m_k' in losses:
                kl_m = torch.stack(losses.kl_m_k, dim=1).mean(dim=0).sum()
            if 'kl_l' in losses:
                kl_l = losses.kl_l.mean(0)
            elif 'kl_l_k' in losses:
                kl_l = torch.stack(losses.kl_l_k, dim=1).mean(dim=0).sum()

            elbo = err + kl_m + kl_l

        return elbo.item()

    def eval(self, inputs):
        output, losses, stats, _, _ = self.model(inputs["input"].to(self.device))
        log_masks = stats["log_m_r_k"]
        ari, _ = average_ari(log_masks, inputs['instances'])
        ari_fg, _ = average_ari(log_masks, inputs['instances'], True)
        mse_batched = ((inputs["input"].cpu()-output.cpu())**2).mean((1, 2, 3)).detach()
        mse = mse_batched.mean(0)
        return ari, ari_fg, mse
    
    def save(self, dir_path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'beta': self.beta,
        }, os.path.join(dir_path, f'ckpt_{self.step}.pt'))

    def restore(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            state_dict = torch.load(f)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.step = state_dict['step']
            if "beta" in state_dict.keys():
                self.beta = state_dict['beta']

    def visualize(self, inputs, writer, mode, iter_idx):
        self.model.eval()
        inputs = inputs.to(self.device)
        output, losses, stats, att_stats, comp_stats = self.model(inputs)

        writer.add_image(mode+'_input', make_grid(inputs), iter_idx)
        writer.add_image(mode+'_recon', make_grid(output), iter_idx)

        ins_seg = torch.argmax(torch.cat(stats["log_m_r_k"], 1), 1, True)
        grid = make_grid(colour_seg_masks(ins_seg))
        writer.add_image(mode+'_instances_r', grid, iter_idx)