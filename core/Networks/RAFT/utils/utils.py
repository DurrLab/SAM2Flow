import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from skimage.morphology import skeletonize
import cv2


def load_ckpt(model, path, device_id=None):
    """ Load checkpoint """
    state_dict = torch.load(path, map_location=torch.device('cpu' if device_id is None else f'cuda:{device_id}'))
    model.load_state_dict(state_dict, strict=False)

def load_encoder_ckpt(model, path, device_id=None):
    """ Load checkpoint for encoder """
    state_dict = torch.load(path, map_location=torch.device('cpu' if device_id is None else f'cuda:{device_id}'))["model"]
    state_dict = {k : v for k, v in state_dict.items() if (k.startswith('fnet.') or k.startswith('channel_convertor.'))}
    model.load_state_dict(state_dict, strict=False)

def resize_data(img1, img2, flow, factor=1.0):
    _, _, h, w = img1.shape
    h = int(h * factor)
    w = int(w * factor)
    img1 = F.interpolate(img1, (h, w), mode='area')
    img2 = F.interpolate(img2, (h, w), mode='area')
    flow = F.interpolate(flow, (h, w), mode='area') * factor
    return img1, img2, flow

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def transform(T, p):
    assert T.shape == (4,4)
    return np.einsum('H W j, i j -> H W i', p, T[:3,:3]) + T[:3, 3]

def from_homog(x):
    return x[...,:-1] / x[...,[-1]]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum('H W, H W j, i j -> H W i', depth1, img_1_coords, np.linalg.inv(K1))
    rel_pose = np.linalg.inv(pose2) @ pose1
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum('H W j, i j -> H W i', cam2_coords, K2))

def induced_flow(depth0, depth1, data):
    H, W = depth0.shape
    coords1 = reproject(depth0, data['T0'], data['T1'], data['K0'], data['K1'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0

    H, W = depth1.shape
    coords1 = reproject(depth1, data['T1'], data['T0'], data['K1'], data['K0'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0
    
    return flow_01, flow_10

def check_cycle_consistency(flow_01, flow_10):
    flow_01 = torch.from_numpy(flow_01).permute(2, 0, 1)[None]
    flow_10 = torch.from_numpy(flow_10).permute(2, 0, 1)[None]
    H, W = flow_01.shape[-2:]
    coords = coords_grid(1, H, W, flow_01.device)
    coords1 = coords + flow_01
    flow_reprojected = bilinear_sampler(flow_10, coords1.permute(0, 2, 3, 1))
    cycle = flow_reprojected + flow_01
    cycle = torch.norm(cycle, dim=1)
    mask = (cycle < 0.1 * min(H, W)).float()
    return mask[0].numpy()

def CEPE(flow, flow_gt, mask):
    """ Compute centerline endpoint error
    Args:
        flow: Predicted optical flow
        flow_gt: Ground truth optical flow
        mask: Binary mask of the intersection between flow and flow_gt
"""
    # # save mask as image
    # cv2.imwrite("mask.png", mask*255)
    # print(mask[mask > 0].shape)
    epe_list = []
    for flo, flo_gt, mask_i in zip(flow, flow_gt, mask):
        # take intersection of flow and flow_gt
        # print(mask_i.shape)
        # # save mask_gt as image
        # cv2.imwrite("mask_gt_raw.png", mask_i.detach().cpu().numpy()*255)
        mask_i = skeletonize(mask_i.detach().cpu().numpy()).astype(np.uint8)

        # # save mask_gt as image
        # cv2.imwrite("mask_gt.png", mask_i*255)
        # # visualize flow_gt
        # flo_gt_vis = flo_gt.permute(1, 2, 0).detach().cpu().numpy()
        # print(flo_gt.shape)
        # flo_gt_vis = flow_to_image(flo_gt_vis, convert_to_bgr=True)
        # cv2.imwrite("flow_gt.png", flo_gt_vis)
        # flo_vis = flo.permute(1, 2, 0).detach().cpu().numpy()
        # flo_vis = flow_to_image(flo_vis, convert_to_bgr=True)
        # cv2.imwrite("flow_prediction.png", flo_vis)

        mask_i = torch.from_numpy(mask_i)
        epe = torch.norm(flo - flo_gt, p=2, dim=0)

        # cv2.imwrite("epe.png", epe.detach().cpu().numpy())
        
        epe = epe[mask_i > 0]
        epe_list.append(epe.mean())
    return torch.stack(epe_list)

def calculate_velocity(flow, flow_gt, mask):
    """ Calculate velocity from optical flow
    Args:
        flow: Optical flow
        flow_gt: Ground truth optical flow
        mask: Binary mask of the object
    """
    flow_vel_list = []
    vel_ratio_list = []
    for flo, flo_gt, mask_i in zip(flow, flow_gt, mask):
        # calculate velocity of flow
        flow_vel = torch.norm(flo, p=2, dim=0)
        # calculate ratio of EPE to velocity
        epe = torch.norm(flo - flo_gt, p=2, dim=0)
        vel_ratio = epe / (flow_vel + 1e-8)
        vel_ratio = vel_ratio[mask_i > 0]
        # calculate mean vel ratio
        vel_ratio_list.append(vel_ratio.mean())
        flow_vel = flow_vel[mask_i > 0]
        flow_vel = flow_vel.mean()
        flow_vel_list.append(flow_vel)
    return torch.stack(flow_vel_list), torch.stack(vel_ratio_list)


def get_metrics(flow_predictions, flow_gt, valid, debug=False):
    """ Compute metrics for optical flow 
    Args:
        flow_predictions: Batched predicted optical flow
        flow_gt: Ground truth optical flow
        valid: Binary mask of the object
    """
    batch_size = flow_predictions.shape[0]
    flow_pr = flow_predictions
    # compute EPE
    epe = torch.norm(flow_pr - flow_gt, p=2, dim=1)
    epe = epe.view(batch_size, -1)

    if debug:
        print(flow_pr.shape)
        print(flow_gt.shape)                  
    # compute intersection between flow and flow_gt
    flow_mask = (torch.norm(flow_pr, p=2, dim=1) > 1) * (torch.norm(flow_gt, p=2, dim=1) > 1)
    # compute CEPE (Centerline Endpoint Error)
    cepe = CEPE(flow_pr, flow_gt, flow_mask)
    # compute FEPE (Foreground EPE)
    fepe = epe.clone()
    fepe[~(flow_mask == (valid >= 0.5)).view(batch_size, -1)] = float('nan')

    epe[~(valid >= 0.5).view(batch_size, -1)] = float('nan')

    flow_vel, epe_2_vel_ratio = calculate_velocity(flow_pr, flow_gt, flow_mask)

    if debug:
        print(flow_mask)
        print(flow_mask.shape)

    epe_non_nan_mask = ~torch.isnan(epe)
    fepe_non_nan_mask = ~torch.isnan(fepe)

    metrics = {
        'epe': epe.nanmean(dim=1).cpu().numpy(),
        '1px': torch.where(epe_non_nan_mask.sum(dim=1) > 0,
                         (((epe < 1) & epe_non_nan_mask).sum(dim=1) / epe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        '3px': torch.where(epe_non_nan_mask.sum(dim=1) > 0,
                         (((epe < 3) & epe_non_nan_mask).sum(dim=1) / epe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        '5px': torch.where(epe_non_nan_mask.sum(dim=1) > 0,
                         (((epe < 5) & epe_non_nan_mask).sum(dim=1) / epe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        "cepe": cepe.cpu().numpy(),
        "fepe": fepe.nanmean(dim=1).cpu().numpy(),
        'f1px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 1) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f3px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 3) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f5px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 5) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f10px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 10) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f15px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 15) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f20px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 20) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f25px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 25) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f30px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 30) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f35px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 35) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        'f40px': torch.where(fepe_non_nan_mask.sum(dim=1) > 0,
                         (((fepe < 40) & fepe_non_nan_mask).sum(dim=1) / fepe_non_nan_mask.sum(dim=1)),
                         torch.tensor(float('nan'))).cpu().numpy(),
        "flow_vel": flow_vel.cpu().numpy(),
        "epe_2_vel_ratio": epe_2_vel_ratio.cpu().numpy()
    }

    return metrics