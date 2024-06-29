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
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from GenesisV2.model import GenesisV2
from GenesisV2.trainer import Trainer

from utils.multi_object_config import load

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
    config_path = 'config/genesisv2.yaml'
    config = load_config(config_path)

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    logdir = config.logging.log_dir + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(results_dir, logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, "config.yaml"), "w") as store_config:
        yaml.dump(config, store_config)

    random.seed(config.general.random_seed)
    np.random.seed(config.general.random_seed)
    torch.manual_seed(config.general.random_seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.general.random_seed)
        print("USING GPU")


    train_loader, val_loader, test_loader = load(config.dataloading)

    model = GenesisV2(config)
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    trainer = Trainer(model, optimizer, device, config)

    if config.training.restore:
        trainer.restore(config.training.restore_path)

    model.to(device)

    writer = SummaryWriter(logdir)

    perm = torch.randperm(config.dataloading.batch_size)
    idx = perm[:config.training.n_samples]
    fixed_imgs = next(iter(val_loader))['input'][idx]



    if config.training.train:
        print("Training")
        start_time = time.time()
        step = 1
        if trainer.step != 0:
                step = trainer.step + 1

        with tqdm(initial=step, total=config.training.epochs) as pbar:
            while step <= config.training.epochs:
                for batch in train_loader:
                    trainer.model.train()
                    err, loss, elbo = trainer.train_step(batch['input'])
                    pbar.set_postfix(loss=loss)

                    if step % config.logging.log_interval == 0:
                        writer.add_scalar('train_loss', loss, step)
                        writer.add_scalar('train_err', err, step)
                        writer.add_scalar('train_elbo', elbo, step)
                        trainer.model.eval()
                        sample_imgs = trainer.visualize(fixed_imgs, writer, 'train', step)
                        #save_image(sample_imgs, os.path.join(logdir, f'slots_at_{step}.jpg'))

                        total_loss = 0
                        for batch in val_loader:
                            loss = trainer.test_step(batch['input'])
                            total_loss += loss
                        test_loss = total_loss / len(val_loader)

                        writer.add_scalar('test_loss', test_loss, step)
                        print(f"Train_loss: {loss:.6f}, Test_loss: {test_loss:.6f}")

                        time_since_start = time.time() - start_time
                        print(f"Time Since Start: {time_since_start:.6f}")

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
        for batch in val_loader:
            loss = trainer.test_step(batch['input'])
            total_loss += loss
        test_loss = total_loss / len(val_loader)
        print(f"Val_loss: {test_loss:.6f}")

    if config.training.visualize:
        print("Visualize Outputs")
        trainer.model.eval()
        batch = next(iter(val_loader))['input'][idx]
        images = trainer.visualize(batch, writer, 'test', 1)
        #save_image(images, os.path.join(logdir, f'outputs_at_test.jpg'))

    if config.training.evaluate:
        prefetch_batches = []
        for i, x in enumerate(test_loader):
            if i == config.general.num_eval_imgs:
                break
            prefetch_batches.append(x)
        model = trainer.model
        model.eval()

        ari, ari_fg, mse = [], [], []
        with torch.no_grad():
            for i, x in enumerate(tqdm(prefetch_batches)):
                ari_, ari_fg_, mse_ = trainer.eval(x)
                ari.append(ari_)
                ari_fg.append(ari_fg_)
                mse.append(mse_)
            
        print(f"Average ARI: {sum(ari)/len(ari)}")
        print(f"Average ARI-FG: {sum(ari_fg)/len(ari_fg)}")
        print(f"Average MSE: {sum(mse)/len(mse)}")

        # dataset_ari(model, test_loader, config.general.num_eval_imgs)


if __name__ == '__main__':
    main()