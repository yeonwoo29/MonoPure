from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
from datasets.dataset.kitti_mmd import KITTIMMD

class MultiPoseDataset(data.Dataset):
  def __init__(self, opt, split, **kwargs):
        self.is_target = kwargs.get('is_target', False)
        self.opt = opt
        self.split = split

        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)


        self._data_rng = np.random.RandomState(123)
        self.max_objs = 128
        self.flip_idx = []
        self.mean = opt.mean if hasattr(opt, 'mean') else np.array([0.408, 0.447, 0.470], dtype=np.float32)
        self.std = opt.std if hasattr(opt, 'std') else np.array([0.289, 0.274, 0.278], dtype=np.float32)
        
        self.annotations = json.load(open(self.ann_path, 'r'))
        self.images = self.annotations 
        
        self.img_dir = os.path.join(opt.data_dir, 'carfusion', 'images', split)
        self.coco = None  # 필요 시 override
        self.num_joints = 17  # 기본값. CarFusion에서 12로 재정의됨
        
  def __len__(self):
    return len(self.images)
  
  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def __getitem__(self, index):
    ann = self.annotations[index]
    img_path = os.path.join(self.img_dir, ann['file_name'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Skipping unreadable image: {img_path}")
        return self.__getitem__((index + 1) % len(self))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width = img.shape[0], img.shape[1]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    rot = 0

    flipped = False
    if self.split == 'train':
        if not self.opt.not_rand_crop:
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
            w_border = self._get_border(128, width)
            h_border = self._get_border(128, height)
            c[0] = np.random.randint(low=w_border, high=width - w_border)
            c[1] = np.random.randint(low=h_border, high=height - h_border)
        else:
            sf = self.opt.scale
            cf = self.opt.shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        if np.random.random() < self.opt.aug_rot:
            rf = self.opt.rotate
            rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
        if np.random.random() < self.opt.flip:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1

    trans_input = get_affine_transform(c, s, rot, [self.opt.input_res, self.opt.input_res])
    inp = cv2.warpAffine(img, trans_input, (self.opt.input_res, self.opt.input_res), flags=cv2.INTER_LINEAR)
    inp = (inp.astype(np.float32) / 255.)
    if self.split == 'train' and not self.opt.no_color_aug:
        color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)

    output_res = self.opt.output_res
    trans_output = get_affine_transform(c, s, 0, [output_res, output_res])
    trans_output_rot = get_affine_transform(c, s, rot, [output_res, output_res])

    hm = np.zeros((self.num_classes, output_res, output_res), dtype=np.float32)
    hm_hp = np.zeros((self.num_joints, output_res, output_res), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    reg = np.zeros((self.max_objs, 2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
    kps = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.float32)
    kp_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)
    hp_offset = np.zeros((self.max_objs * self.num_joints, 2), dtype=np.float32)
    hp_ind = np.zeros((self.max_objs * self.num_joints), dtype=np.int64)
    hp_mask = np.zeros((self.max_objs * self.num_joints), dtype=np.int64)

    ####
    hp_offset_vec = hp_offset[:self.num_joints].reshape(1, -1)




    # bbox를 [x1, y1, x2, y2] -> [x, y, w, h]
    bbox = np.array([
        affine_transform([ann['bbox'][0], ann['bbox'][1]], trans_output),
        affine_transform([ann['bbox'][2], ann['bbox'][3]], trans_output)
    ], dtype=np.float32).reshape(-1)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_res - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_res - 1)

    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ct = np.array([cx, cy], dtype=np.float32)
    ct_int = ct.astype(np.int32)

    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
    radius = max(0, int(radius))
    draw_umich_gaussian(hm[0], ct_int, radius)

    wh[0] = w, h
    ind[0] = ct_int[1] * output_res + ct_int[0]
    reg[0] = ct - ct_int
    reg_mask[0] = 1

    keypoints = np.array(ann['keypoints'], dtype=np.float32).reshape(self.num_joints, 3)
    for j in range(self.num_joints):
        x, y, v = keypoints[j]
        if v > 0:
            pt = affine_transform([x, y], trans_output_rot)
            if 0 <= pt[0] < output_res and 0 <= pt[1] < output_res:
                kps[0, j * 2: j * 2 + 2] = pt[0] - ct_int[0], pt[1] - ct_int[1]
                kp_mask[0, j * 2: j * 2 + 2] = 1
                pt_int = pt.astype(np.int32)
                
                idx = 0 * self.num_joints + j  # 0: 첫 객체 index
                hp_offset[idx] = pt - pt_int
                hp_ind[idx] = pt_int[1] * output_res + pt_int[0]
                hp_mask[idx] = 1

                hp_ind[j] = pt_int[1] * output_res + pt_int[0]
                hp_mask[j] = 1
                draw_umich_gaussian(hm_hp[j], pt_int, radius)

    ret = {
        'input': torch.from_numpy(inp).float(),
        'hm': torch.from_numpy(hm).float(),
        'wh': torch.from_numpy(wh).float(),
        'reg': torch.from_numpy(reg).float(),
        'ind': torch.from_numpy(ind).long(),
        'reg_mask': torch.from_numpy(reg_mask),
        'hps': torch.from_numpy(kps).float(),
        'hps_mask': torch.from_numpy(kp_mask),
        'hm_hp': torch.from_numpy(hm_hp).float(),
        #'hp_offset': torch.from_numpy(hp_offset[:self.num_joints]).float(),  # (12, 2)
        'hp_offset': torch.from_numpy(hp_offset_vec).float(),
        
        
        'hp_ind': torch.from_numpy(hp_ind[:self.num_joints]).long(),
        'hp_mask': torch.from_numpy(hp_mask[:self.num_joints])

    
    }
    return ret

