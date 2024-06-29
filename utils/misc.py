import tensorflow as tf
import time
import torch
import json 
from sklearn.metrics import adjusted_rand_score
import numpy as np
import datetime

tf.compat.v1.disable_eager_execution()


def len_tfrecords(dataset, sess):
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) #dataset.make_one_shot_iterator()
    frame = iterator.get_next()
    total_sz = 0
    while True:
        try:
            _ = sess.run(frame)
            total_sz += 1
            if total_sz % 1000 == 0:
                print(total_sz)
        except tf.errors.OutOfRangeError:
            return total_sz


# def np_img_centre_crop(np_img, crop_dim, batch=False):
#     # np_img: [c, dim1, dim2] if batch == False else [batch_sz, c, dim1, dim2]
#     shape = np_img.shape
#     if batch:
#         s2 = (shape[2]-crop_dim)//2
#         s3 = (shape[3]-crop_dim)//2
#         return np_img[:, :, s2:s2+crop_dim, s3:s3+crop_dim]
#     else:
#         s1 = (shape[1]-crop_dim)//2
#         s2 = (shape[2]-crop_dim)//2
#         return np_img[:, s1:s1+crop_dim, s2:s2+crop_dim]


def loader_throughput(loader, num_batches=100, burn_in=5):
    assert num_batches > 0
    if burn_in is None:
        burn_in = num_batches // 10
    num_samples = 0
    print(f"Train loader throughput stats on {num_batches} batches...")
    for i, batch in enumerate(loader):
        if i == burn_in:
            timer = time.time()
        if i >= burn_in:
            num_samples += batch['input'].size(0)
        if i == num_batches + burn_in:
            break
    dt = time.time() - timer
    spb = dt / num_batches
    ips = num_samples / dt
    print(f"{spb:.3f} s/b, {ips:.1f} im/s")

def colour_seg_masks(masks, palette='15'):
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)
    assert masks.dim() == 4
    assert masks.shape[1] == 1
    colours = json.load(open(f'/app/utils/colour_palette{palette}.json'))
    img_r = torch.zeros_like(masks)
    img_g = torch.zeros_like(masks)
    img_b = torch.zeros_like(masks)
    for c_idx in range(masks.max().item() + 1):
        c_map = masks == c_idx
        if c_map.any():
            img_r[c_map] = colours['palette'][c_idx][0]
            img_g[c_map] = colours['palette'][c_idx][1]
            img_b[c_map] = colours['palette'][c_idx][2]
    return torch.cat([img_r, img_g, img_b], dim=1)

# def log_scalars(sdict, tag, step, writer):
#     for key, val in sdict.items():
#         writer.add_scalar(f'{tag}/{key}', val, step)

# def iou_binary(mask_A, mask_B):
#     assert mask_A.shape == mask_B.shape
#     assert mask_A.dtype == torch.bool
#     assert mask_B.dtype == torch.bool
#     intersection = (mask_A * mask_B).sum((1, 2, 3))
#     union = (mask_A + mask_B).sum((1, 2, 3))
#     # Return -100 if union is zero, else return IOU
#     return torch.where(union == 0, torch.tensor(-100.0),
#                        intersection.float() / union.float())

# def average_segcover(segA, segB, ignore_background=False):
#     """
#     Covering of segA by segB
#     segA.shape = [batch size, 1, img_dim1, img_dim2]
#     segB.shape = [batch size, 1, img_dim1, img_dim2]

#     scale: If true, take weighted mean over IOU values proportional to the
#            the number of pixels of the mask being covered.

#     Assumes labels in segA and segB are non-negative integers.
#     Negative labels will be ignored.
#     """

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
#     assert segA.shape[1] == 1 and segB.shape[1] == 1
#     bsz = segA.shape[0]
#     nonignore = (segA >= 0)

#     mean_scores = torch.tensor(bsz*[0.0], device=device)
#     N = torch.tensor(bsz*[0], device=device)
#     scaled_scores = torch.tensor(bsz*[0.0], device=device)
#     scaling_sum = torch.tensor(bsz*[0], device=device)

#     # Find unique label indices to iterate over
#     if ignore_background:
#         iter_segA = torch.unique(segA[segA > 0]).tolist()
#     else:
#         iter_segA = torch.unique(segA[segA >= 0]).tolist()
#     iter_segB = torch.unique(segB[segB >= 0]).tolist()
#     # Loop over segA
#     for i in iter_segA:
#         binaryA = segA == i
#         if not binaryA.any():
#             continue
#         max_iou = torch.tensor(bsz*[0.0], device=device)
#         # Loop over segB to find max IOU
#         for j in iter_segB:
#             # Do not penalise pixels that are in ignore regions
#             binaryB = (segB == j) * nonignore
#             if not binaryB.any():
#                 continue
#             iou = iou_binary(binaryA, binaryB)
#             max_iou = torch.where(iou > max_iou, iou, max_iou)
#         # Accumulate scores
#         mean_scores += max_iou
#         N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
#         scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
#         scaling_sum += binaryA.sum((1, 2, 3))

#     # Compute coverage
#     mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float()
#     scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

#     # Sanity check
#     assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
#     assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
#     assert (mean_scores[N == 0] == 0).all()
#     assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
#     assert (scaled_scores[N == 0] == 0).all()
#     assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

#     # Return mean over batch dimension 
#     return mean_sc.mean(0), scaled_sc.mean(0)


def average_ari(log_m_k, instances, foreground_only=False):
    ari = []
    masks_stacked = torch.stack(log_m_k, dim=4).exp().detach()
    masks_split = torch.split(masks_stacked, 1, dim=0)
    # Loop over elements in batch
    for i, m in enumerate(masks_split):
        masks_pred = np.argmax(m.cpu().numpy(), axis=-1).flatten()
        masks_gt = instances[i].detach().cpu().numpy().flatten()
        if foreground_only:
            masks_pred = masks_pred[np.where(masks_gt > 0)]
            masks_gt = masks_gt[np.where(masks_gt > 0)]
        score = adjusted_rand_score(masks_pred, masks_gt)
        ari.append(score)
    return sum(ari)/len(ari), ari


# def dataset_ari(model, data_loader, num_images=300):

#     model.eval()

#     print("Computing ARI on dataset")
#     ari = []
#     ari_fg = []
#     model.eval()
#     for bidx, batch in enumerate(data_loader):
#         if next(model.parameters()).is_cuda:
#             batch['input'] = batch['input'].cuda()
#         with torch.no_grad():
#             _, _, stats, _, _ = model(batch['input'])

#         # Return zero if labels or segmentations are not available
#         if 'instances' not in batch or not hasattr(stats, 'log_m_k'):
#             return 0., 0., [0], [0]

#         _, ari_list = average_ari(stats.log_m_k, batch['instances'])
#         _, ari_fg_list = average_ari(stats.log_m_k, batch['instances'], True)
#         ari += ari_list
#         ari_fg += ari_fg_list
#         if bidx % 1 == 0:
#             log_ari = sum(ari)/len(ari)
#             log_ari_fg = sum(ari_fg)/len(ari_fg)
#             t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             print(f"{t} | After [{len(ari)} / {num_images}] images: " +
#                    f"ARI {log_ari:.4f}, FG ARI {log_ari_fg:.4f}")
#         if len(ari) >= num_images:
#             break

#     assert len(ari) == len(ari_fg)
#     ari = ari[:num_images]
#     ari_fg = ari_fg[:num_images]

#     avg_ari = sum(ari)/len(ari)
#     avg_ari_fg = sum(ari_fg)/len(ari_fg)
#     print(f"FINAL ARI for {len(ari)} images: {avg_ari:.4f}")
#     print(f"FINAL FG ARI for {len(ari_fg)} images: {avg_ari_fg:.4f}")

#     model.train()

#     return avg_ari, avg_ari_fg, ari_list, ari_fg_list