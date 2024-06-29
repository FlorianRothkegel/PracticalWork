import os
import time
import random
import numpy as np
from tqdm import tqdm
import yaml

import warnings

warnings.filterwarnings("ignore", message=".*CuDNN issue.*")
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from SlotAttention.model import SlotAttentionModel
from SlotAttention.trainer import Trainer

from utils.multi_object_config import load

from utils.segmentation_metrics import adjusted_rand_index

import tensorflow as tf

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

    def __repr__(self):
        return repr(self.__dict__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)

def main():

    config_path = 'config/slot_attention.yaml'
    config = load_config(config_path)


    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')

    if not (os.path.exists(results_dir)):
        os.makedirs(results_dir)

    logdir = config.logging.log_dir + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(results_dir, logdir)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    with open(os.path.join(logdir, "config.yaml"), "w") as store_config:
        yaml.dump(config, store_config)
    

    random.seed(config.general.random_seed)
    np.random.seed(config.general.random_seed)
    torch.manual_seed(config.general.random_seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
        torch.cuda.manual_seed(config.general.random_seed)
    else:
        device = torch.device('cpu')

    train_loader, val_loader, test_loader = load(config.dataloading)

    model = SlotAttentionModel(
                    resolution=(config.dataloading.img_size, config.dataloading.img_size),
                    num_slots = config.model.num_slots,
                    num_iters = config.model.num_slots_iters,
                    device= device,
                    in_channels =3,
                    num_hidden = 4,
                    hdim = 32,
                    slot_size = 64,
                    slot_mlp_size = 128,
                    decoder_resolution=(config.dataloading.img_size, config.dataloading.img_size)
                    )
    optimizer = optim.Adam(model.parameters(), lr = config.training.learning_rate)

    trainer = Trainer(model, optimizer, device)
    writer = SummaryWriter(logdir)

    if config.training.restore:
        trainer.restore(config.training.restore_path)

    perm = torch.randperm(config.dataloading.batch_size)
    idx = perm[: config.training.n_samples]
    fixed_imgs = (next(iter(val_loader))['input'])[idx] 

    if config.training.train:

        print("Training")
        start_time = time.time()
        step=1
        if trainer.step !=0:
            step = trainer.step + 1
        
        with tqdm(initial=step, total=config.training.epochs) as pbar:
            while step <= config.training.epochs:
                for batch in train_loader:

                    trainer.model.train()
                    if step < config.training.warmup_steps:
                        learning_rate = config.training.learning_rate * (step/ config.training.warmup_steps)
                    else:
                        learning_rate = config.training.learning_rate

                    learning_rate = learning_rate * (config.training.decay_rate ** (
                        step / config.training.decay_steps))

                    trainer.optimizer.param_groups[0]['lr'] = learning_rate

                    loss = trainer.train_step(batch['input'])
                    pbar.set_postfix(loss=loss)

                    if step % config.logging.log_interval == 0:
                        writer.add_scalar('train_loss', loss.item(), step)
                        trainer.model.eval()
                        sample_imgs = trainer.visualize(fixed_imgs)
                        writer.add_image(f'slots at epoch {step}', sample_imgs, step)
                        save_image(sample_imgs, os.path.join(logdir, f'slots_at_{step}.jpg'))

                        total_loss = 0
                        for batch in val_loader:
                            loss = trainer.test_step(batch['input'])
                            total_loss += loss.item()
                        test_loss = total_loss/len(val_loader)

                        writer.add_scalar('val_loss',test_loss, step)

                        print("###############################")
                        print(f"At training step {step}")
                        print("###############################")
                        print(f"Train_loss: {loss.item():.6f}")
                        print(f"Val_loss: {test_loss:.6f}")   

                        time_since_start = time.time() - start_time
                        print(f"Time Since Start {time_since_start:.6f}")

                    if step % config.training.ckpt_interval == 0:
                        trainer.save(logdir)

                    step += 1
                    pbar.update(1)
                    trainer.step = step
                    if step > config.training.epochs:
                        break
         
    if config.training.test:
        print("Testing")
        trainer.model.eval()
        total_loss = 0
        for batch in test_loader:
            loss = trainer.test_step(batch['input'])
            total_loss += loss.item()
        test_loss = total_loss/len(val_loader)
        print(f"Val_loss: {test_loss:.6f}")

    if config.training.visualize:
        print("Visualize Slots")
        trainer.model.eval()
        perm = torch.randperm(config.dataloading.batch_size)
        idx = perm[: config.training.n_samples]
        batch = next(iter(val_loader))['input'][idx]
        images = trainer.visualize(batch)
        save_image(images, os.path.join(logdir, f'slots_at_test.jpg'))
        

    if config.training.evaluate:
        prefetch_batches = []
        for i, x in enumerate(test_loader):
            if i == config.general.num_eval_imgs:
                break
            prefetch_batches.append(x)
        model = trainer.model
        model.eval()

        ari_list = []
        mse_list = []
        with torch.no_grad():
            for i, x in enumerate(tqdm(prefetch_batches)):
                recon_combined, recon, masks, slots = model(x['input'].to(device))
                desired_shape = [1, config.dataloading.img_size * config.dataloading.img_size, config.model.num_slots]
                gt_masks = x["instances"].permute(0, 4, 2, 3, 1).contiguous()
                gt_masks = gt_masks.view(desired_shape).cpu()
                pred_masks = masks.permute(0,2,3,4,1).contiguous()
                pred_masks = pred_masks.view(desired_shape).cpu()
                ari_test = adjusted_rand_index(gt_masks[...,1:], pred_masks)

                mse_batched = ((x["input"].cpu()-recon_combined.cpu())**2).mean((1, 2, 3)).detach()
                mse = mse_batched.mean(0)
                mse_list.append(mse)

                with tf.compat.v1.Session() as sess:
                    ari = sess.run(ari_test)
                    ari_list.append(ari)


        print(f"Average ARI: {sum(ari_list)/len(ari_list)}")
        print(f"Average MSE: {sum(mse_list)/len(mse_list)}")

if __name__ == '__main__':
    main()