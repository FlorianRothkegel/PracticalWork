import os
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid


class Trainer:

    def __init__(self, model, optimizer, device):

        self.model  = model
        self.optimizer = optimizer
        self.loss = nn.MSELoss()
        self.device = device
        self.model.to(self.device)
        self.step = 0

    def train_step(self, inputs):

        inputs = inputs.to(self.device)
        recon_combined, recon, masks, slots = self.model(inputs)
        loss = self.loss(recon_combined, inputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def test_step(self, inputs):

        inputs = inputs.to(self.device)
        with torch.no_grad():
            recon_combined, recon, masks, slots = self.model(inputs)
            loss = self.loss(recon_combined, inputs)
        return loss

    def visualize(self, inputs):

        inputs = inputs.to(self.device)
        with torch.no_grad():
            recon_combined, recon, masks, slots = self.model(inputs)

        out = torch.cat(
                [
                    inputs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recon * masks + (1 - masks),  # each slot
                ],
                dim=1,
            )
        batch_size, num_slots, C, H, W = recon.shape
        images = make_grid(
            out.view(batch_size * out.shape[1], C, H, W).cpu(), normalize=False, nrow=out.shape[1],
        )

        return images

    def save(self, dir_path):

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step
            }, os.path.join(dir_path, f'ckpt_{self.step}.pt'))

    def restore(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            state_dict = torch.load(f)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.step = state_dict['step']