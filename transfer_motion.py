from tqdm import trange
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import Logger
from modules.discriminator import LipDiscriminator, NoiseDiscriminator
from modules.discriminator import Encoder
from utils.util import landmarkdict_to_mesh_tensor, mesh_tensor_to_landmarkdict, LIP_IDX, ROI_IDX, get_seg, draw_mesh_images, interpolate_zs, init_dir
from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from torch.utils.data import Dataset
import argparse
import os
import shutil
import numpy as np
import random
import pickle as pkl
import math
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, default='data/sonny/test.mp4')
parser.add_argument('--driving_dir', type=str, default='data/sonny/train.mp4')


args = parser.parse_args()

ROI_DIM = 3 * len(ROI_IDX)

roi_dict_dir = os.path.join(args.data_dir, 'roi_dict_normalized')
if os.path.exists(roi_dict_dir):
    shutil.rmtree(roi_dict_dir)

init_dir(roi_dict_dir)

def get_mapping_function(roi_pool, roi_drving_pool):
    # roi_ref, roi_driving_ref: roi_dim
    # roi_pool: P x roi_dim
    # roi_driving_pool: P' x roi_dim
    # all normalized by [-1, 1]

    A = torch.ones_like(roi_pool[0])
    B = torch.zeros_like(roi_pool[0])

    # tgt_max, tgt_min = roi_pool.max(dim=0)[0], roi_pool.min(dim=0)[0]
    # src_max, src_min = roi_driving_pool.max(dim=0)[0], roi_driving_pool.min(dim=0)[0]
    # A = (tgt_max - tgt_min) / (src_max - src_min).clamp(min=1e-6)
    # B = tgt_min - A * src_min

    # var, mean = torch.var_mean(roi_pool, dim=0, unbiased=False)
    # var *= 3
    # var_driving, mean_driving = torch.var_mean(roi_driving_pool, dim=0, unbiased=True)
    # mean_driving = roi_driving_pool[0]
    # # var, mean: lip_dim
    # A = (var / var_driving.clamp(min=1e-6)).sqrt() # lip_dim
    # B = mean - A * mean_driving # lip_dim
    print('constructing mapping function: A={}, B={}'.format(A, B))
    return lambda x: A * x + B


roi_pool = torch.load(os.path.join(args.data_dir, 'mesh_stack.pt'))[:, ROI_IDX].flatten(1) / 128
roi_driving_pool = torch.load(os.path.join(args.driving_dir, 'mesh_stack.pt'))[:, ROI_IDX].flatten(1) / 128

mapping_func = get_mapping_function(roi_pool, roi_driving_pool)


num_frames = min(len(roi_driving_pool), len(roi_pool))
roi_driving_pool = roi_driving_pool[:num_frames]

for i, roi in enumerate(tqdm(roi_driving_pool)):
    # roi: roi_dim
    key = '{:05d}'.format(i + 1)
    mapped_roi = mapping_func(roi)  # roi_dim
    torch.save(mapped_roi.view(-1, 3) * 128, os.path.join(roi_dict_dir, key + '.pt'))




    

