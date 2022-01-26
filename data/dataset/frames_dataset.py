import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from .augmentation import AllAugmentationWithMeshTransform
import glob
import random
import pickle as pkl
from datetime import datetime
import utils.util as util

def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class MeshFramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), is_train=True,
                 random_seed=0, id_sampling=False, pairs_list=None, augmentation_params=None, num_dummy_set=0):
        self.root_dir = root_dir
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.num_dummy_set = num_dummy_set
        # if os.path.exists(os.path.join(root_dir, 'train')):
        #     assert os.path.exists(os.path.join(root_dir, 'test'))
        #     print("Use predefined train-test split.")
        #     train_videos = os.listdir(os.path.join(root_dir, 'train'))
        #     test_videos = os.listdir(os.path.join(root_dir, 'test'))
        #     self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        # else:
        #     print("Use random train-test split.")
        #     train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        self.videos = list(filter(lambda x: x.endswith('.mp4'), os.listdir(root_dir)))
        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationWithMeshTransform(**augmentation_params)
        else:
            self.transform = None
        
        self.length = {}
        print('Dataset size: {}'.format(self.__len__()))


    def __len__(self):
        length = 0
        for vid in self.videos:
            path = os.path.join(self.root_dir, vid)
            num_frames = len(os.listdir(os.path.join(path, 'img')))
            length += num_frames
            self.length[vid] = num_frames
        return length

    def __getitem__(self, idx):
        name = random.choice(self.videos)
        idx %= self.length[name]
        path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)
    
        frames = sorted(os.listdir(os.path.join(path, 'img')))
        num_frames = len(frames)
        frame_idx = [(idx + int(datetime.now().timestamp())) % self.length[name], idx] if self.is_train else range(min(500, num_frames))

        mesh_dicts = [torch.load(os.path.join(path, 'mesh_dict', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        mesh_dicts_normed = [torch.load(os.path.join(path, 'mesh_dict_normalized', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        R_array = [np.array(mesh_dict['R']) for mesh_dict in mesh_dicts]
        t_array = [np.array(mesh_dict['t']) for mesh_dict in mesh_dicts]
        c_array = [np.array(mesh_dict['c']) for mesh_dict in mesh_dicts]
        mesh_array = [np.array(list(mesh_dict.values())[:478]) for mesh_dict in mesh_dicts]
        normed_mesh_array = [np.array(list(mesh_dict_normed.values())[:478]) for mesh_dict_normed in mesh_dicts_normed]
        z_array = [torch.load(os.path.join(path, 'z', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        normed_z_array = [torch.load(os.path.join(path, 'z_normalized', frames[frame_idx[i]].replace('.png', '.pt'))) for i in range(len(frame_idx))]
        video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[frame_idx[i]]))) for i in range(len(frame_idx))]
        mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image', frames[frame_idx[i]]))) for i in range(len(frame_idx))]

        R_array.append(R_array[1])
        t_array.append(t_array[1])
        c_array.append(c_array[1])
        mesh_array.append(mesh_array[1])
        normed_mesh_array.append(normed_mesh_array[1])
        video_array.append(video_array[1])
        mesh_img_array.append(mesh_img_array[1])
        z_array.append(z_array[1])
        normed_z_array.append(normed_z_array[1])

        if self.transform is not None:
            video_array, mesh_array, R_array, t_array, c_array, mesh_img_array = self.transform(video_array, mesh_array, R_array, t_array, c_array, mesh_img_array)

        video_array = np.array(video_array, dtype='float32')
        if self.is_train:
            mesh_img_array = np.array(mesh_img_array, dtype='float32')
            normed_z_array = torch.stack(normed_z_array, dim=0).float() / 128 - 1
        mesh_array = np.array(mesh_array, dtype='float32') / 128 - 1
        normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
        R_array = np.array(R_array, dtype='float32')
        c_array = np.array(c_array, dtype='float32') * 128
        t_array = np.array(t_array, dtype='float32')
        t_array = t_array + np.matmul(R_array, (c_array[:, None, None] * np.ones_like(t_array)))
        z_array = torch.stack(z_array, dim=0).float() / 128 - 1

        out = {}

        source = video_array[0]
        real = video_array[1]
        driving = video_array[2]
        source_mesh = mesh_array[0]
        real_mesh = mesh_array[1]
        driving_mesh = mesh_array[2]
        source_normed_mesh = normed_mesh_array[0]
        real_normed_mesh = normed_mesh_array[1]
        driving_normed_mesh = normed_mesh_array[2]
        source_R = R_array[0]
        real_R = R_array[1]
        driving_R = R_array[2]
        source_t = t_array[0]
        real_t = t_array[1]
        driving_t = t_array[2]
        source_c = c_array[0]
        real_c = c_array[1]
        driving_c = c_array[2]
        source_mesh_image = mesh_img_array[0]
        real_mesh_image = mesh_img_array[1]
        driving_mesh_image = mesh_img_array[2]
        # source_mesh_image = mesh_img_array[0] * lip_mask_array[0]
        # real_mesh_image = mesh_img_array[1] * lip_mask_array[1]
        # driving_mesh_image = mesh_img_array[2] * lip_mask_array[2]
        source_z = z_array[0]
        real_z = z_array[1]
        driving_z = z_array[2]
        source_normed_z = normed_z_array[0]
        real_normed_z = normed_z_array[1]
        driving_normed_z = normed_z_array[2]

        out['driving'] = driving.transpose((2, 0, 1))
        out['real'] = real.transpose((2, 0, 1))
        out['source'] = source.transpose((2, 0, 1))
        out['driving_mesh'] = {'mesh': driving_mesh, 'normed_mesh': driving_normed_mesh, 'R': driving_R, 't': driving_t, 'c': driving_c, 'z': driving_z, 'normed_z': driving_normed_z}
        out['real_mesh'] = {'mesh': real_mesh, 'normed_mesh': real_normed_mesh, 'R': real_R, 't': real_t, 'c': real_c, 'z': real_z, 'normed_z': real_normed_z}
        out['source_mesh'] = {'mesh': source_mesh, 'normed_mesh': source_normed_mesh, 'R': source_R, 't': source_t, 'c': source_c, 'z': source_z, 'normed_z': source_normed_z}
        out['driving_mesh_image'] = driving_mesh_image.transpose((2, 0, 1))
        out['real_mesh_image'] = real_mesh_image.transpose((2, 0, 1))
        out['source_mesh_image'] = source_mesh_image.transpose((2, 0, 1))
        out['name'] = video_name

        return out
