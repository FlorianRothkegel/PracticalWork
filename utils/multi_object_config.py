
import torch
import torch.nn.functional as F
import tensorflow as tf

import numpy as np
from .misc import loader_throughput, len_tfrecords

from .multi_dsprites import dataset as multi_dspritesDataset
from .tetrominoes import dataset as tetrominoesDataset



MULTI_DSPRITES = '/multi_dsprites_multi_dsprites_colored_on_grayscale.tfrecords'
TETROMINOS = '/tetrominoes_tetrominoes_train.tfrecords'

SEED = 0


def load(cfg, **unused_kwargs):
    global SEED
    SEED = cfg.seed
    tf.random.set_seed(SEED)

    del unused_kwargs
    print(f"Using {cfg.num_workers} data workers.")

    sess = tf.compat.v1.InteractiveSession()

    if cfg.dataset == 'multi_dsprites':
        cfg.img_size = 64 if cfg.img_size < 0 else cfg.img_size
        cfg.K_steps = 6 if cfg.K_steps < 0 else cfg.K_steps
        background_entities = 1
        max_frames = 1000000
        raw_dataset = multi_dspritesDataset(
            cfg.data_folder + MULTI_DSPRITES,
            'colored_on_grayscale',
            map_parallel_calls=cfg.num_workers if cfg.num_workers > 0 else None)
    elif cfg.dataset == 'tetrominoes':
        cfg.img_size = 32 if cfg.img_size < 0 else cfg.img_size
        cfg.K_steps = 4 if cfg.K_steps < 0 else cfg.K_steps
        background_entities = 1
        max_frames = 1000000
        raw_dataset = tetrominoesDataset(
            cfg.data_folder + TETROMINOS,
            map_parallel_calls=cfg.num_workers if cfg.num_workers > 0 else None)
    else:
        raise NotImplementedError(f"{cfg.dataset} not a valid dataset.")

    # Split into train / val / test
    if cfg.dataset_size > max_frames:
        print(f"WARNING: {cfg.dataset_size} frames requested, "\
                "but only {max_frames} available.")
        cfg.dataset_size = max_frames
    if cfg.dataset_size > 0:
        total_sz = cfg.dataset_size
        raw_dataset = raw_dataset.take(total_sz)
    else:
        total_sz = max_frames
    if total_sz < 0:
        print("Determining size of dataset...")
        total_sz = len_tfrecords(raw_dataset, sess)
    print(f"Dataset has {total_sz} frames")
    
    val_sz = 10000
    tst_sz = 10000
    tng_sz = total_sz - val_sz - tst_sz
    assert tng_sz > 0
    print(f"Splitting into {tng_sz}/{val_sz}/{tst_sz} for tng/val/tst")
    tst_dataset = raw_dataset.take(tst_sz)
    val_dataset = raw_dataset.skip(tst_sz).take(val_sz)
    tng_dataset = raw_dataset.skip(tst_sz + val_sz)

    tng_loader = MultiOjectLoader(sess, tng_dataset, background_entities,
                                  tng_sz, cfg.batch_size,
                                  cfg.img_size, cfg.buffer_size, genesis=cfg.genesis)
    val_loader = MultiOjectLoader(sess, val_dataset, background_entities,
                                  val_sz, cfg.batch_size,
                                  cfg.img_size, cfg.buffer_size, genesis=cfg.genesis)
    tst_loader = MultiOjectLoader(sess, tst_dataset, background_entities,
                                  tst_sz, 1,
                                  cfg.img_size, cfg.buffer_size, genesis=cfg.genesis)

    if not cfg.debug:
        loader_throughput(tng_loader)

    return (tng_loader, val_loader, tst_loader)


class MultiOjectLoader():

    def __init__(self, sess, dataset, background_entities,
                 num_frames, batch_size, img_size=64, buffer_size=128, genesis=False):
        dataset = dataset.shuffle(buffer_size*batch_size, seed=SEED)
        dataset = dataset.batch(batch_size)
        self.dataset = dataset.prefetch(buffer_size)
        self.sess = sess
        self.background_entities = background_entities
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.length = self.num_frames // batch_size
        self.img_size = img_size
        self.count = 0
        self.frames = None
        self.genesis = genesis

    def __len__(self):
        return self.length

    def __iter__(self):
        it = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        self.frames = it.get_next()
        return self

    def __next__(self):
        try:
            frame = self.sess.run(self.frames)
            self.count += 1

            img = frame['image']
            img = np.moveaxis(img, 3, 1)
            shape = img.shape
            img = torch.FloatTensor(img) / 255.
            if self.img_size != shape[2]:
                img = F.interpolate(img, size=self.img_size)

            
            if not self.genesis:
                masks = torch.FloatTensor(frame['mask'])
                if self.img_size != shape[2]:
                    masks = F.interpolate(masks.squeeze(-1), size=self.img_size)
                    masks = masks.unsqueeze(4)
            else:
                raw_masks = frame['mask']
                masks = np.zeros((shape[0], 1, shape[2], shape[3]), dtype='int')

                cond = np.where(raw_masks[:, :, :, :, 0] == 255, True, False)
                # Ignore background entities
                num_entities = cond.shape[1]
                for o_idx in range(self.background_entities, num_entities):
                    masks[cond[:, o_idx:o_idx+1, :, :]] = o_idx + 1
                masks = torch.FloatTensor(masks)
                if self.img_size != shape[2]:
                    masks = F.interpolate(masks, size=self.img_size)
                masks = masks.type(torch.LongTensor)

            return {'input': img, 'instances': masks}

        except tf.errors.OutOfRangeError:
            self.count = 0
            raise StopIteration
