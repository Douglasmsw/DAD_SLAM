import torch
import render_rays
import torch.nn.functional as F
import gc

def step_batch_loss(alpha, color, gt_depth, gt_color, sem_labels, mask_depth, z_vals,
                    color_scaling=5.0, opacity_scaling=10.0, clip_model=None, 
                    semantic_loss=False, semantic_scaling=1, norm='L1', preprocess=None, 
                    full_alpha = None,
                    full_gt_color=None, full_color=None,  full_mask_depth=None): # BRAD & DOUG
    """
    apply depth where depth are valid                                       -> mask_depth
    apply depth, color loss on this_obj & unkown_obj == (~other_obj)        -> mask_obj
    apply occupancy/opacity loss on this_obj & other_obj == (~unknown_obj)  -> mask_sem

    output:
    loss for training
    loss_all for per sample, could be used for active sampling, replay buffer
    """
    mask_obj = sem_labels != 0
    mask_obj = mask_obj.detach()
    mask_sem = sem_labels != 2
    mask_sem = mask_sem.detach()


    alpha = alpha.squeeze(dim=-1)
    color = color.squeeze(dim=-1)


    occupancy = render_rays.occupancy_activation(alpha)
    termination = render_rays.occupancy_to_termination(occupancy, is_batch=True)  # shape [num_batch, num_ray, points_per_ray]

    render_depth = render_rays.render(termination, z_vals)
    diff_sq = (z_vals - render_depth[..., None]) ** 2
    var = render_rays.render(termination, diff_sq).detach()  # must detach here!
    render_color = render_rays.render(termination[..., None], color, dim=-2)
    render_opacity = torch.sum(termination, dim=-1)     # similar to obj-nerf opacity loss

    # 2D depth loss: only on valid depth & mask
    # [mask_depth & mask_obj]
    # loss_all = torch.zeros_like(render_depth)
    loss_depth_raw = render_rays.render_loss(render_depth, gt_depth, loss="L1", normalise=False)
    loss_depth = torch.mul(loss_depth_raw, mask_depth & mask_obj)   # keep dim but set invalid element be zero
    # loss_all += loss_depth
    loss_depth = render_rays.reduce_batch_loss(loss_depth, var=var, avg=True, mask=mask_depth & mask_obj)   # apply var as imap

    # 2D color loss: only on obj mask
    # [mask_obj]
    loss_col_raw = render_rays.render_loss(render_color, gt_color, loss="L1", normalise=False) 
    loss_col = torch.mul(loss_col_raw.sum(-1), mask_obj)
    # loss_all += loss_col / 3. * color_scaling
    loss_col = render_rays.reduce_batch_loss(loss_col, var=None, avg=True, mask=mask_obj)
    
    # BRAD & DOUG
    if semantic_loss:
        full_alpha = full_alpha.squeeze(dim=-1)
        # batches channels height widthy
        full_occupancy = render_rays.occupancy_activation(full_alpha)
        full_termination = render_rays.occupancy_to_termination(full_occupancy, is_batch=True) 


        render_pred = render_rays.render(full_termination[..., None], full_color, dim=-2)
        

        loss_sem_raw = render_rays.render_semantic_loss(clip_model, loss=norm, normalize=False, preprocess=preprocess,
                                                        full_gt_color=full_gt_color, full_color=render_pred, full_mask_depth=full_mask_depth)
        loss_sem = loss_sem_raw.sum(-1)
        loss_sem = render_rays.reduce_batch_loss(loss_sem, var=None, avg=True, mask=mask_obj)

    # 2D occupancy/opacity loss: apply except unknown area
    # [mask_sem]
    # loss_opacity_raw = F.mse_loss(torch.clamp(render_opacity, 0, 1), mask_obj.float().detach()) # encourage other_obj to be empty, while this_obj to be solid
    # print("opacity max ", torch.max(render_opacity.max()))
    # print("opacity min ", torch.max(render_opacity.min()))
    loss_opacity_raw = render_rays.render_loss(render_opacity, mask_obj.float(), loss="L1", normalise=False)
    loss_opacity = torch.mul(loss_opacity_raw, mask_sem)  # but ignore -1 unkown area e.g., mask edges
    # loss_all += loss_opacity * opacity_scaling
    loss_opacity = render_rays.reduce_batch_loss(loss_opacity, var=None, avg=True, mask=mask_sem)   # todo var

    if semantic_loss:
        # loss for bp
        l_batch = loss_depth + loss_col * color_scaling + loss_opacity * opacity_scaling + loss_sem * semantic_scaling

    else:
        # loss for bp
        l_batch = loss_depth + loss_col * color_scaling + loss_opacity * opacity_scaling
   
    # loss for bp
    l_batch = loss_depth + loss_col * color_scaling + loss_opacity * opacity_scaling
    loss = l_batch.sum()

    return loss, None       # return loss, loss_all.detach()
