import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch import sparse_coo_tensor
import numpy as np

from .utils.utils import bilinear_sampler

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    print("[!!alt_cuda_corr is not compiled!!]")
    pass


class DirectCorr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords):
        ctx.save_for_backward(fmap1, fmap2, coords)
        corr, = alt_cuda_corr.forward(fmap1, fmap2, coords, 4)
        return corr

    def backward(ctx, grad_output):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            alt_cuda_corr.backward(fmap1, fmap2, coords, grad_output, 4)

        return fmap1_grad, fmap2_grad, coords_grad


class OLCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        batch, dim, ht, wd = fmap1.shape
        self.fmap1 = fmap1.permute(0, 2, 3, 1).view(batch*ht*wd, 1, dim)

        self.fmap2_pyramid = []
        self.fmap2_pyramid.append(fmap2)
        for i in range(self.num_levels - 1):
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.fmap2_pyramid.append(fmap2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        _, _, dim = self.fmap1.shape

        out_pyramid = []
        for i in range(self.num_levels):
            fmap2 = self.fmap2_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1, indexing='ij').to(coords.device)

            centroid_lvl = coords.reshape(batch, h1 * w1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 1, (2 * r + 1) ** 2, 2)
            coords_lvl = centroid_lvl + delta_lvl

            fmap2 = bilinear_sampler(fmap2, coords_lvl) # B, 256, h*w, 9*9
            fmap2 = fmap2.permute(0, 2, 1, 3).view(batch*h1*w1, dim, (2 * r + 1) ** 2)
            #print(self.fmap1.shape, fmap2.shape)
            corr = torch.bmm(self.fmap1, fmap2) / torch.sqrt(torch.tensor(dim).float())

            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).float()


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            # print(corr.shape, coords_lvl.shape)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
    

class CorrBlock_Context:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock_Context.corr(fmap1, fmap2)
        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            self.corr_pyramid.append(F.avg_pool2d(self.corr_pyramid[-1], 2, stride=2))
        
        # Precompute delta
        r = self.radius    
        dx, dy = torch.meshgrid(
            torch.linspace(-r, r, 2 * r + 1),
            torch.linspace(-r, r, 2 * r + 1),
            indexing="ij"
        )
        self.delta_lvl = torch.stack((dy, dx), dim=-1).view(1, 2 * r + 1, 2 * r + 1, 2).to(corr.device)
        fbatch, _, h1, w1 = fmap1.shape
        self.corr_out = torch.empty(fbatch * h1 * w1, 1, 2*r+1, 2*r+1, device=corr.device)

        self.weights = None
        self.indices = None
    
    def __call__(self, coords):
        if self.weights is None or self.indices is None:
           raise NotImplementedError("No ROI added yet")
        
        coords = coords.permute(0, 2, 3, 1) # batch, h1, w1, 2
        batch, h1, w1, _ = coords.shape
        coords = coords.reshape(batch, h1 * w1, 1, 1, 2)
        locations = []
        for obj_id, coord in enumerate(coords):
            location = coord[self.indices[self.batch_id[obj_id]:self.batch_id[obj_id+1]]]
            locations.append(location) 
        locations = torch.cat(locations, dim=0)

        out = torch.zeros((batch, self.num_levels, h1*w1, 1, 2*self.radius+1, 2*self.radius+1), device=coords.device)
        self.corr_out.zero_().to(coords.dtype)
        for i, corr_ori in enumerate(self.corr_pyramid):
            centroid_lvl = locations / (2 ** i)
            coords_lvl = centroid_lvl + self.delta_lvl
            corr = bilinear_sampler(corr_ori, coords_lvl)
            for obj_id in range(batch):
                self.corr_out.zero_()
                inds = self.indices[self.batch_id[obj_id]:self.batch_id[obj_id+1]]
                corr_obj = corr[self.batch_id[obj_id]:self.batch_id[obj_id+1]]
                expanded_indices =inds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, corr_obj.size(1), corr_obj.size(2), corr_obj.size(3))
                self.corr_out.scatter_add_(0, expanded_indices, corr_obj)
                # (h1xw1) x x 1 x 2r+1 x 2r+1
                print("corr_out:", self.corr_out.shape)
                print(out[obj_id, i].shape)
                out[obj_id, i] = self.corr_out
        # B x num_levels  x (h1xw1) x 1 x 2r+1 x 2r+1 => B x (h1xw1) x (num_levelsx(2r+1)x(2r+1))
        out = out.permute(0,2,1,3,4,5).flatten(start_dim=2)
        # B x (num_levelsx(2r+1)x(2r+1)) x h1 x w1
        C = out.shape[-1]
        out = out.view(batch, h1, w1, C).permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        fbatch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(fbatch, dim, ht * wd)
        fmap2 = fmap2.view(fbatch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(fbatch * ht * wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
    
    def add_context(self, roi, threshold=0.5):
        batch, _, h1, w1= roi.shape
        self.weights, self.indices, self.batch_id = self.getforeground(roi, threshold)

        for i, corr in enumerate(self.corr_pyramid):
            corr_roi = corr[self.indices] * self.weights
            self.corr_pyramid[i] = corr_roi

        foreground_area_sum = self.batch_id[-1].item()
        # print(f"Foreground area: {foreground_area_sum / (h1 * w1)}")
        return foreground_area_sum / (h1 * w1)
    
    def getforeground(self, roi, threshold):
        """
        Compute differentiable pixel locations above a given threshold.
        roi: B, 1, H, W
        """
        # soft binarization
        # threshold = threshold #torch.mean(roi, axis=(1,2,3), keepdim=True)
        # roi = torch.sigmoid((roi - threshold) * 10)
        roi = torch.sigmoid(roi)
        if threshold < 0:
            roi = torch.ones_like(roi)
        # weights: B x (HW)
        weights = roi.flatten(1)
        indices = [torch.where(weight > 0.5)[0] for weight in weights]
        weights_idxed = [weight[ind][:, None, None, None] for weight, ind in zip(weights, indices)]
        bid = np.cumsum([0,]+[len(ind) for ind in indices])

        indices = torch.cat(indices, dim=0)
        weights_idxed = torch.cat(weights_idxed, dim=0)
        # print(indices.shape, weights_idxed.shape, bid)
        return weights_idxed, indices, bid
    
class CorrBlock_Context_batch:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # fbatch * (HW), 1, H, W
        corr = CorrBlock_Context_batch.corr(fmap1, fmap2)
        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            self.corr_pyramid.append(F.avg_pool2d(self.corr_pyramid[-1], 2, stride=2))
        
        # Precompute delta
        r = self.radius    
        dx, dy = torch.meshgrid(
            torch.linspace(-r, r, 2 * r + 1),
            torch.linspace(-r, r, 2 * r + 1),
            indexing="ij"
        )
        self.delta_lvl = torch.stack((dy, dx), dim=-1).view(1, 2 * r + 1, 2 * r + 1, 2).to(corr.device)
        fbatch, _, h1, w1 = fmap1.shape
        self.corr_out = torch.empty(fbatch * h1 * w1, 1, 2*r+1, 2*r+1, device=corr.device)

        self.weights = None
        self.indices = None
    
    def __call__(self, coords):
        if self.weights is None or self.indices is None:
           raise NotImplementedError("No ROI added yet")
        
        coords = coords.permute(0, 2, 3, 1) # batch, h1, w1, 2
        batch, h1, w1, _ = coords.shape
        coords = coords.reshape(batch, h1 * w1, 1, 1, 2)
        locations = []
        for obj_id, coord in enumerate(coords):
            location = coord[self.indices[self.batch_id[obj_id]:self.batch_id[obj_id+1]]]
            locations.append(location) 
        locations = torch.cat(locations, dim=0)

        out = torch.zeros((batch, self.num_levels, h1*w1, 1, 2*self.radius+1, 2*self.radius+1), device=coords.device)
        self.corr_out.zero_().to(coords.dtype)
        for i, corr_ori in enumerate(self.corr_pyramid):
            centroid_lvl = locations / (2 ** i)
            coords_lvl = centroid_lvl + self.delta_lvl
            corr = bilinear_sampler(corr_ori, coords_lvl)
            for obj_id in range(batch):
                self.corr_out.zero_()
                inds = self.indices[self.batch_id[obj_id]:self.batch_id[obj_id+1]]
                corr_obj = corr[self.batch_id[obj_id]:self.batch_id[obj_id+1]]
                expanded_indices =inds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, corr_obj.size(1), corr_obj.size(2), corr_obj.size(3))
                self.corr_out.scatter_add_(0, expanded_indices, corr_obj)
                # (h1xw1) x x 1 x 2r+1 x 2r+1
                print("corr_out:", self.corr_out.shape)
                print(out[obj_id, i].shape)
                out[obj_id, i] = self.corr_out
        # B x num_levels  x (h1xw1) x 1 x 2r+1 x 2r+1 => B x (h1xw1) x (num_levelsx(2r+1)x(2r+1))
        out = out.permute(0,2,1,3,4,5).flatten(start_dim=2)
        # B x (num_levelsx(2r+1)x(2r+1)) x h1 x w1
        C = out.shape[-1]
        out = out.view(batch, h1, w1, C).permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        fbatch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(fbatch, dim, ht * wd)
        fmap2 = fmap2.view(fbatch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(fbatch * ht * wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
    
    def add_context(self, roi, threshold=0.5):
        batch, _, h1, w1= roi.shape
        self.weights, self.indices, self.batch_id = self.getforeground(roi, threshold)

        for i, corr in enumerate(self.corr_pyramid):
            corr_roi = corr[self.indices] * self.weights
            self.corr_pyramid[i] = corr_roi

        foreground_area_sum = self.batch_id[-1].item()
        # print(f"Foreground area: {foreground_area_sum / (h1 * w1)}")
        return foreground_area_sum / (h1 * w1)
    
    def getforeground(self, roi, threshold):
        """
        Compute differentiable pixel locations above a given threshold.
        roi: B, 1, H, W
        """
        # soft binarization
        # threshold = threshold #torch.mean(roi, axis=(1,2,3), keepdim=True)
        # roi = torch.sigmoid((roi - threshold) * 10)
        roi = torch.sigmoid(roi)
        if threshold < 0:
            roi = torch.ones_like(roi)
        # weights: B x (HW)
        weights = roi.flatten(1)
        indices = [torch.where(weight > 0.5)[0] for weight in weights]
        weights_idxed = [weight[ind][:, None, None, None] for weight, ind in zip(weights, indices)]
        bid = np.cumsum([0,]+[len(ind) for ind in indices])

        indices = torch.cat(indices, dim=0)
        weights_idxed = torch.cat(weights_idxed, dim=0)
        # print(indices.shape, weights_idxed.shape, bid)
        return weights_idxed, indices, bid
        


class CorrBlock_attended:
    def __init__(self, fmap1, fmap2, att, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # Precompute all pairs correlation
        self.weights, self.indices = self.getforeground(att)
        corr = CorrBlock_attended.corr(fmap1, fmap2, self.indices)
        corr = corr * self.weights[:, None, None, None]

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            self.corr_pyramid.append(F.avg_pool2d(self.corr_pyramid[-1], 2, stride=2))
        
        # Precompute delta
        r = self.radius    
        dx, dy = torch.meshgrid(
            torch.linspace(-r, r, 2 * r + 1),
            torch.linspace(-r, r, 2 * r + 1),
            indexing="ij"
        )
        self.delta_lvl = torch.stack((dy, dx), dim=-1).view(1, 2 * r + 1, 2 * r + 1, 2).to(corr.device)

        batch, _, h1, w1 = fmap1.shape
        self.corr_out = torch.empty(len(self.corr_pyramid), batch * h1 * w1, 1, 2*r+1, 2*r+1, device=corr.device) #corr[:,:,:2*r+1,:2*r+1]
        # self.corr_out = torch.zeros(len(self.corr_pyramid), batch * h1 * w1, 1, 2*r+1, 2*r+1, device=corr.device)
    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1) # batch, h1, w1, 2
        batch, h1, w1, _ = coords.shape

        locations = coords.reshape(batch * h1 * w1, 1, 1, 2)[self.indices] * self.weights[:,None,None,None]
        
        out_pyramid = []
        for i, corr in enumerate(self.corr_pyramid):
            ## with or without * weights
            centroid_lvl = locations / 2 ** i # coords.reshape(batch * h1 * w1, 1, 1, 2)[indices]
            coords_lvl = centroid_lvl + self.delta_lvl
            # print(corr.shape, coords_lvl.shape)
            corr = bilinear_sampler(corr, coords_lvl)
            # corr_out = torch.zeros((batch * h1 * w1,)+corr.shape[1:], device=corr.device)
            expanded_indices = self.indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, corr.size(1), corr.size(2), corr.size(3))
            # corr_out.scatter_add_(0, expanded_indices, corr)
            # out_pyramid.append(corr_out.view(batch, h1, w1, -1))
            if i == 0:
                self.corr_out.zero_().to(corr.dtype)
            self.corr_out[i].scatter_add_(0, expanded_indices, corr)
            # self.out_pyramid
            # corr_out = self.corr_out.view(batch, h1, w1, -1)
        # out = torch.cat(out_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()

        out = self.corr_out.permute(1,0,2,3,4).flatten(start_dim=1)
        out = out.view(batch, h1, w1, -1).permute(0, 3, 1, 2).contiguous().float()
        
        return out, self.weights.shape[0] / (batch * h1 * w1)

    @staticmethod
    def corr(fmap1, fmap2, indices):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch * ht * wd, 1, ht, wd)[indices]
        return corr / torch.sqrt(torch.tensor(dim).float())
    
    def getforeground(self, att):
        """
        Compute differentiable pixel locations above a given threshold.
        att: b, H, W, 1
        """
        # threshold = torch.mean(att, axis=(1,2,3), keepdim=True)
        # weights = torch.sigmoid((att - threshold) * 10)
        weights = att.flatten()
        indices = torch.where(weights > 0.5)[0]
        # mask = (weights > 0.5).float()
        # indices = mask.nonzero(as_tuple=True)[0]
        return weights[indices], indices


class CorrBlockSingleScale(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()
        self.radius = radius

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1, indexing='ij').to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            #fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((None, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous().float()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous().float()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = DirectCorr.apply(fmap1_i, fmap2_i, coords_i)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
