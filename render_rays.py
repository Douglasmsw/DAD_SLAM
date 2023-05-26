import torch
import numpy as np
import clip
import torchvision
import sys
from torchvision.utils import save_image
import gc

# clip_model, _ = clip.load(args.clip_model, device=cfg.training_device, jit=args.jit)
# preprocess = Compose([
#     Resize(size=224, interpolation=Image.BICUBIC, max_size=None, antialias=False),
#     CenterCrop(size=(224, 224)),
#     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
#         ])

def occupancy_activation(alpha, distances=None):
    # occ = 1.0 - torch.exp(-alpha * distances)
    # print("applying sigmoid activation")
    occ = torch.sigmoid(alpha)    # unisurf

    return occ

def alpha_to_occupancy(depths, dirs, alpha, add_last=False):
    interval_distances = depths[..., 1:] - depths[..., :-1]
    if add_last:
        last_distance = torch.empty(
            (depths.shape[0], 1),
            device=depths.device,
            dtype=depths.dtype).fill_(0.1)
        interval_distances = torch.cat(
            [interval_distances, last_distance], dim=-1)

    dirs_norm = torch.norm(dirs, dim=-1)
    interval_distances = interval_distances * dirs_norm[:, None]
    occ = occupancy_activation(alpha, interval_distances)

    return occ

def occupancy_to_termination(occupancy, is_batch=False):
    if is_batch:
        first = torch.ones(list(occupancy.shape[:2]) + [1], device=occupancy.device)
        free_probs = (1. - occupancy + 1e-10)[:, :, :-1]
    else:
        first = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
        free_probs = (1. - occupancy + 1e-10)[:, :-1]
    free_probs = torch.cat([first, free_probs], dim=-1)
    term_probs = occupancy * torch.cumprod(free_probs, dim=-1)

    # using escape probability
    # occupancy = occupancy[:, :-1]
    # first = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
    # free_probs = (1. - occupancy + 1e-10)
    # free_probs = torch.cat([first, free_probs], dim=-1)
    # last = torch.ones([occupancy.shape[0], 1], device=occupancy.device)
    # occupancy = torch.cat([occupancy, last], dim=-1)
    # term_probs = occupancy * torch.cumprod(free_probs, dim=-1)

    return term_probs

def render(termination, vals, dim=-1):
    weighted_vals = termination * vals
    render = weighted_vals.sum(dim=dim)

    return render

def render_loss(render, gt, loss="L1", normalise=False):

    residual = render - gt
    if loss == "L2":
        loss_mat = residual ** 2
    elif loss == "L1":
        loss_mat = torch.abs(residual)
    else:
        print("loss type {} not implemented!".format(loss))

    if normalise:
        loss_mat = loss_mat / gt

    return loss_mat

def render_semantic_loss(clip_model, loss="L1", normalize=False, preprocess=None,
                        full_gt_color=None, full_color=None, full_mask_depth=None):  # 3, X, Y

    xdim, ydim = full_gt_color.size()[-2], full_gt_color.size()[-1]
    full_color = full_color.reshape(shape = (full_color.shape[0], xdim, ydim, full_color.shape[2]))
    full_color = full_color.permute(0, 3, 1, 2)

    assert full_gt_color.size() == full_color.size()

    if full_mask_depth != None:
        full_color = full_color * full_mask_depth
        full_gt_color = full_gt_color * full_mask_depth
    full_gt_color = full_gt_color / 255
    full_color = preprocess(full_color)
    full_gt_color = preprocess(full_gt_color)


    render_emb = clip_model.encode_image(full_color)
    gt_emb = clip_model.encode_image(full_gt_color)

    #gc.collect()
    #torch.cuda.empty_cache()



    residual = torch.sub(gt_emb, render_emb)

    if loss == "L2":
        loss_mat = residual ** 2
    elif loss == "L1":
        loss_mat = torch.abs(residual)
    else:
        print("loss type {} not implemented!".format(loss))

    if normalize:
        loss_mat = loss_mat / gt_emb
    
    #gc.collect()
    #torch.cuda.empty_cache()

    return loss_mat

def reduce_batch_loss(loss_mat, var=None, avg=True, mask=None, loss_type="L1"):
    mask_num = torch.sum(mask, dim=-1)
    if (mask_num == 0).any():   # no valid sample, return 0 loss
        loss = torch.zeros_like(loss_mat)
        if avg:
            loss = torch.mean(loss, dim=-1)
        return loss
    if var is not None:
        eps = 1e-4
        if loss_type == "L2":
            information = 1.0 / (var + eps)
        elif loss_type == "L1":
            information = 1.0 / (torch.sqrt(var) + eps)

        loss_weighted = loss_mat * information
    else:
        loss_weighted = loss_mat

    if avg:
        if mask is not None:
            loss = (torch.sum(loss_weighted, dim=-1)/(torch.sum(mask, dim=-1)+1e-10))
            if (loss > 100000).any():
                print("loss explode")
                exit(-1)
        else:
            loss = torch.mean(loss_weighted, dim=-1).sum()
    else:
        loss = loss_weighted

    return loss

def make_3D_grid(occ_range=[-1., 1.], dim=256, device="cuda:0", transform=None, scale=None):
    print("Initializing linespace")
    t = torch.linspace(occ_range[0], occ_range[1], steps=dim, device=device)
    print("Initializing meshgrid")
    grid = torch.meshgrid(t, t, t)
 
    print("Building 3D grid")
    grid_3d = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
    )

    if scale is not None:
        print("Scaling 3D grid")
        grid_3d = grid_3d * scale
    if transform is not None:
        print("Transforming 3D grid")
        print("Building transformation matrix rows")
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        print("Building transformation matrix grids")
        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)
        print("Concatenating transformation matrix rows into 3D tensor")
        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)
        
        trans = transform[None, None, None, :3, 3]
        print("Adding transform to 3D transformation tensor")
        grid_3d = grid_3d + trans

    print("Transformation grid complete\nReturning grid_3d")
    # print(torch.cuda.memory_summary())
    return grid_3d
