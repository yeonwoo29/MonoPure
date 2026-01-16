import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class KITTIMMD(Dataset):
    def __init__(self, opt, split='train'):
        self.opt = opt
        self.split = split

        self.img_dir = os.path.join(opt.data_dir, 'kittimmd', 'images', split)
        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, '*')))

        self.input_h = opt.input_h
        self.input_w = opt.input_w
        self.mean = opt.mean
        self.std = opt.std

    def __len__(self):
        return len(self.img_paths)

    def _transform(self, img):
        img = cv2.resize(img, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return img

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._transform(img)

        return {'input': torch.from_numpy(img).float()}
