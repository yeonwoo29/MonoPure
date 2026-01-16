from __future__ import absolute_import, division, print_function
import os
import json
import cv2
import numpy as np
import torch
import math

from datasets.sample.multi_pose import MultiPoseDataset
from utils.image import get_affine_transform, affine_transform
from utils.image import draw_umich_gaussian, gaussian_radius

class CarFusion(MultiPoseDataset):
    num_classes = 1
    num_joints = 12
    default_resolution = [512, 512]
    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32)
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32)
    flip_idx = []

    def __init__(self, opt, split='train'):
        self.data_dir = os.path.join(opt.data_dir, 'carfusion')
        self.img_dir = os.path.join(self.data_dir, 'images', split)
        self.ann_path = os.path.join(self.data_dir, 'annotations', f'{split}.json')

        # ✅ 먼저 ann_path 설정 후 super 호출
        super(CarFusion, self).__init__(opt, split)

        print(f"✅ Loading {split} annotations from {self.ann_path}")
        with open(self.ann_path, 'r') as f:
            self.annotations = json.load(f)

        self.images = self.annotations
        self.num_joints = CarFusion.num_joints
