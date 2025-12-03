import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import random
from glob import glob
import os.path as osp
from torch.utils.data.distributed import DistributedSampler

from ..utils import frame_utils
from ..utils.augmentor_video import FlowAugmentor, SparseFlowAugmentor
from ..utils.utils import forward_interpolate
from .datasets_video import FlowDataset, FlowDatasetTest
from scipy.ndimage import distance_transform_edt
import pickle
import cv2
from skimage.morphology import erosion, dilation, disk

# spring
class CapFlow_SAM_stable(FlowDataset):
    def __init__(self, 
                 root_path= '/data/512_stable',
                 raw_root = '/data/512_stable_raw', 
                 split='Train',
                 aug_params=None, 
                 input_frames=5, 
                 forward_warp=False,
                 debug = False,
                 gen_pts = False):
        super(CapFlow_SAM_stable, self).__init__(aug_params=aug_params, 
                                                 input_frames=input_frames, 
                                                 forward_warp=forward_warp,
                                                 sparse=False, 
                                                 subsample_groundtruth=False)
        video_list = glob(osp.join(root_path, split)+'/*')
        # exclude files that are not directories
        video_list = [v for v in video_list if osp.isdir(v)]
        self.mask_list = []
        masks = {}
        root_raw = osp.join(raw_root, split)
        mask_imgs = sorted(next(os.walk(osp.join(root_raw, "mask")))[2])
        for mask in mask_imgs:
            masks[osp.basename(mask).split('.')[0]] = osp.join(root_raw, "mask", mask)

        for video in video_list:
            flow_path = sorted(glob(os.path.join(video, 'Flow_float16')+'/*.npy'))
            for fi in range(0, len(flow_path) - input_frames + 1, input_frames-1):
                flow_seg = [flow_path[fi + i] for i in range(input_frames - 1)]
                self.flow_list.append(flow_seg)
                fnums = [int(os.path.basename(f).split('.')[0]) for f in flow_seg]
                fnums.append(fnums[-1] + 1)
                self.image_list.append([os.path.join(video, '{:04d}.png'.format(i)) for i in fnums])
                self.has_gt_list.append([True] * (input_frames - 1))
                self.mask_list.append(masks[osp.basename(video)])
        print(f'[dataset size: {len(self.image_list)}]')
        self.debug = debug
        self.gen_pts = gen_pts
        # print(self.image_list[0:3])
        # print(self.flow_list[0:3])
        # print(self.has_gt_list[0:3])

    def gen_random_pt(self, mask, pt_num = 3, debug = False):
        """
        Generate random points from the mask
        mask: binary mask
        pt_num: the number of points to generate
        """
        # cast mask to binary
        mask = mask > 0
        pts = {}
        # get foreground points 
        fg = distance_transform_edt(mask)
        pdf = fg / np.sum(fg)

        if debug:
            debug_out = {}
            debug_out['pdf'] = pdf
            debug_out['fg'] = fg

        pdf_flat = pdf.flatten()
        idx_fg = np.random.choice(pdf_flat.size, p=pdf_flat, size=pt_num*10)
        idx_fg.sort()
        idx_fg = idx_fg[::10]
        # Convert the flat index back into 2D coordinates
        point_y, point_x = np.unravel_index(idx_fg, pdf.shape)
        FG_points = np.stack([point_y, point_x]).T
        
        bg = ~mask
        # add border of 0 on the mask 
        border_mask = np.zeros_like(bg)
        border_mask[1:-1, 1:-1] = 1
        bg = bg * border_mask
        if debug:
            debug_out['border_mask'] = border_mask
            debug_out['bg'] = bg
        # get background points away from the fuzzy edge
        bg = distance_transform_edt(bg > 0) ** 2
        pdf_bg = bg / np.sum(bg)
        pdf_bg_flat = pdf_bg.flatten()
        idx_bg = np.random.choice(pdf_bg_flat.size, p=pdf_bg_flat, size=pt_num*10)
        idx_bg.sort()
        idx_bg = idx_bg[::10]
        point_y_bg, point_x_bg = np.unravel_index(idx_bg, pdf_bg.shape)
        BG_points = np.stack([point_y_bg, point_x_bg]).T

        pts['points'] = np.concatenate([FG_points, BG_points], axis = 0)
        pts['labels'] = np.concatenate([np.ones(pt_num), np.zeros(pt_num)])
        if debug:
            debug_out['FG_points'] = FG_points
            debug_out['BG_points'] = BG_points
            debug_out['pdf_bg'] = pdf_bg
            return pts, debug_out
        return pts

    def gen_random_pt1(self, mask, img = None, pt_num = 3):
        if img is not None:
            img_value = np.sum(img,axis = 2)
            valid_mask = (img_value > 0).astype(np.float32)
        else:
            valid_mask = np.ones_like(mask).astype(np.float32)
        
        # get foreground points 
        fg_mask = mask > 0 
        fg_mask = erosion(fg_mask, disk(5))
        fg = distance_transform_edt(fg_mask)
        fg = fg * valid_mask
        
        pdf = fg / np.sum(fg)
        pdf_flat = pdf.flatten()
        index = np.random.choice(pdf_flat.size, p=pdf_flat, size=pt_num)
        # Convert the flat index back into 2D coordinates
        point_y, point_x = np.unravel_index(index, pdf.shape)
        FG_points = np.stack([point_y, point_x]).T
        
        # get background points away from the fuzzy edge
        bg_mask = (mask == 0) * valid_mask        
        bg_mask = erosion(bg_mask, disk(10))
        bg = distance_transform_edt(bg_mask)
        bg = bg * valid_mask
        
        pdf_bg = bg / np.sum(bg)
        pdf_bg_flat = pdf_bg.flatten()
        idx_bg = np.random.choice(pdf_bg_flat.size, p=pdf_bg_flat, size=pt_num)
        point_y_bg, point_x_bg = np.unravel_index(idx_bg, pdf_bg.shape)
        BG_points = np.stack([point_y_bg, point_x_bg]).T
        
        pts = np.concatenate([FG_points, BG_points], axis = 0)
        labels = np.concatenate([np.ones((len(FG_points),)), np.zeros((len(FG_points),))])
        return {'points': pts, 'labels': labels}
        
        

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valids = None
        flows = [np.load(path).astype(np.float32) for path in self.flow_list[index]]
        
        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]
        imgs = [np.array(img).astype(np.float32) for img in imgs]
        # # nromalize the image
        # # get 5 percentile intensity value of non-zero pixels
        # min_val = np.percentile(np.concatenate([img[img > 0] for img in imgs]), 0.5)
        # max_val = np.percentile(np.concatenate([img[img > 0] for img in imgs]), 99.5)
        # imgs = [(img - min_val) / (max_val - min_val) for img in imgs]
        # imgs = [np.clip(img, 0, 1) * 255 for img in imgs]
        imgs = [img.astype(np.uint8) for img in imgs]
        
        mask = cv2.imread(self.mask_list[index], cv2.IMREAD_GRAYSCALE)

        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        if self.augmentor is not None:
            if self.sparse:
                imgs, flows, valids = self.augmentor(imgs, flows, valids)
            else:
                imgs, flows = self.augmentor(imgs, flows)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        masks = torch.from_numpy(mask).unsqueeze(0).repeat_interleave(self.input_frames,0).float()
        
        if valids is None:
            valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
            o_valids = False
        else:
            valids = [torch.from_numpy(valid).float() for valid in valids]
            o_valids = True
        
        # if self.subsample_groundtruth:
        #     flows = [flow[::2, ::2] for flow in flows]
        if not self.gen_pts:
            return torch.stack(imgs), torch.stack(flows), masks, torch.stack(valids)
        
        # choose a random number between 1 and 3
        num_labels = np.random.randint(1, 4)
        if self.debug:
            debug_list = []
        pts_list = []
        for i in range(num_labels):
            # out = self.gen_random_pt(mask, pt_num = 3, debug = self.debug)
            out = self.gen_random_pt1(mask, imgs[i], pt_num = 3)
            if self.debug:
                pts, debug_output = out
                debug_list.append(debug_output)
                pts_list.append(pts)
            else:
                pts = out
                pts_list.append(pts)
        
        pts_dict = {}
        pts_dict['points'] = torch.stack([torch.tensor(pt['points']) for pt in pts_list])
        pts_dict['labels'] = torch.stack([torch.tensor(pt['labels']) for pt in pts_list])

        if self.debug:
            return torch.stack(imgs), torch.stack(flows), torch.stack(valids), pts_dict, debug_list
        return torch.stack(imgs), torch.stack(flows), masks, torch.stack(valids), pts_dict
        # else:
        #     new_size = (flows[0].shape[1] // 8, flows[0].shape[2] // 8)
        #     if not o_valids:
        #         downsampled_flow = [F.interpolate(flow.unsqueeze(0), size=new_size, mode='bilinear', align_corners=True).squeeze(0) / 8 for flow in flows[:-1]]
        #         forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] + [forward_interpolate(flow) for flow in downsampled_flow]
        #     else:
        #         forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] * len(flows)
        #     if self.debug:
        #         return torch.stack(imgs), torch.stack(flows), torch.stack(valids), torch.stack(forward_warped_flow), pts_dict, debug_list
        #     return torch.stack(imgs), torch.stack(flows), torch.stack(valids), torch.stack(forward_warped_flow), pts_dict
        

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H', DDP=False, rank=0):
    forward_warp = 'concat_flow' in args[args.network] and args[args.network].concat_flow
    if args.dataset == 'stable':
        # aug_params = {'crop_size': args.image_size, 'min_scale': 0, 'max_scale': 0.4, 'do_flip': True}
        aug_params = None
        train_dataset  = Dataset_SAM_stable(aug_params=aug_params, input_frames=args.input_frames, forward_warp=forward_warp)
    else:
        raise NotImplementedError('Dataset not supported: ' + args.dataset)
    
    print('Training with %d image pairs' % len(train_dataset))
    if DDP:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size,
                                           rank=rank, shuffle=True)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size // args.world_size,
                                       pin_memory=True, shuffle=False, num_workers=16, sampler=train_sampler)
        return train_sampler, train_loader
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

        return train_loader