import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte, img_as_float32, io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

from modules.generator import MeshOcclusionAwareGenerator
from utils.util import mesh_tensor_to_landmarkdict, draw_mesh_images, interpolate_zs, mix_mesh_tensor, LIP_IDX, ROI_IDX, MASK_IDX, WIDE_MASK_IDX, get_lip_mask, init_dir
from utils.one_euro_filter import OneEuroFilter

from scipy.spatial import ConvexHull
import os
import shutil
import ffmpeg
import cv2
import pickle as pkl
from utils.logger import Visualizer

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    # xs_hat = applyFilter(xs, t, 0.005, 0.7)
    # ys_hat = applyFilter(ys, t, 0.005, 0.7, mouthPoints + chins)
    # ys_hat = applyFilter(ys_hat, t, 0.000001, 1.5, rest)
    # zs_hat = applyFilter(zs, t, 0.005, 0.7)

MIN_CUTOFF = 0.005
BETA = 0.7

def preprocess_mesh(m, frame_idx):
    roi = ROI_IDX
    res = m.copy()
    for key in res.keys():
        # print('{} shape: {}'.format(key, torch.tensor(res[key][frame_idx]).shape))
        res[key] = torch.tensor(res[key][frame_idx])[None].float().cuda()
    # print('raw shape: {}'.format(res['normed_mesh'].shape))
    if 'normed_roi' in res:
        res['mesh'][:, roi] = res['normed_roi']
        res['normed_mesh'][:, roi] = res['normed_roi']
    res['value'] = res['normed_mesh'][:, :, :2]

    return res

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = MeshOcclusionAwareGenerator(**config['model_params']['generator_params'], **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    d = generator.state_dict()
    d.update(checkpoint['generator'])
    generator.load_state_dict(d)
    
    if not cpu:
        generator = DataParallelWithCallback(generator)

    generator.eval()
    
    return generator

def get_dataset(path):
    video_name = os.path.basename(path)
    frames = sorted(os.listdir(os.path.join(path, 'img')))
    num_frames = min(len(frames), len(os.listdir(os.path.join(path, 'roi_dict_normalized'))))
    frames = frames[:num_frames]
    frame_idx = range(num_frames)

    reference_frame_path = os.path.join(path, 'frame_reference.png')
    reference_mesh_img_path = os.path.join(path, 'mesh_image_reference.png')
    reference_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference.pt'))
    reference_normed_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference.pt'))
    reference_frame = img_as_float32(io.imread(reference_frame_path))
    reference_mesh_img = img_as_float32(io.imread(reference_mesh_img_path))
    reference_mesh = np.array(list(reference_mesh_dict.values())[:478])
    reference_normed_mesh = np.array(list(reference_normed_mesh_dict.values())[:478])
    reference_R = np.array(reference_mesh_dict['R'])
    reference_t = np.array(reference_mesh_dict['t'])
    reference_c = np.array(reference_mesh_dict['c'])
    reference_normed_z = torch.load(os.path.join(path, 'z_reference_normalized.pt'))
    video_array = [reference_frame for idx in frame_idx]
    mesh_img_array = [reference_mesh_img for idx in frame_idx]
    mesh_array = [reference_mesh for idx in frame_idx]
    normed_mesh_array = [reference_normed_mesh for idx in frame_idx]
    z_array = [reference_normed_z for idx in frame_idx]
    R_array = [reference_R for idx in frame_idx]
    t_array = [reference_t for idx in frame_idx]
    c_array = [reference_c for idx in frame_idx]

    mesh_dict = 'mesh_dict'
    normed_mesh_dict = 'mesh_dict_normalized'
    roi_dict = 'roi_dict_normalized'
    driving_roi_array = [torch.load(os.path.join(path, roi_dict, frames[idx].replace('.png', '.pt'))).cpu().detach().numpy() for idx in frame_idx]
    
    _mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
    _normed_mesh_array = [np.array(list(torch.load(os.path.join(path, normed_mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
    _R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
    _t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
    _c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]

    video_array = np.array(video_array, dtype='float32')
    mesh_img_array = np.array(mesh_img_array, dtype='float32')
    mesh_array = np.array(mesh_array, dtype='float32') / 128 - 1
    normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
    R_array = np.array(R_array, dtype='float32')
    c_array = np.array(c_array, dtype='float32') * 128
    t_array = np.array(t_array, dtype='float32')
    t_array = t_array + np.matmul(R_array, (c_array[:, None, None] * np.ones_like(t_array)))
    z_array = torch.stack(z_array, dim=0).float() / 128 - 1
    out = {}

    driving_roi_array = np.array(driving_roi_array, dtype='float32') / 128 - 1

    _mesh_array = np.array(_mesh_array, dtype='float32') / 128 - 1
    _normed_mesh_array = np.array(_normed_mesh_array, dtype='float32') / 128 - 1
    _R_array = np.array(_R_array, dtype='float32')
    _c_array = np.array(_c_array, dtype='float32') * 128
    _t_array = np.array(_t_array, dtype='float32')
    _t_array = _t_array + np.matmul(_R_array, (_c_array[:, None, None] * np.ones_like(_t_array)))

    video = video_array
    out['video'] = video.transpose((3, 0, 1, 2))
    out['mesh'] = {'mesh': mesh_array, 'normed_mesh': normed_mesh_array, 'R': R_array, 't': t_array, 'c': c_array, 'normed_z': z_array, 'normed_roi': driving_roi_array}
    out['driving_mesh_img'] = mesh_img_array.transpose((3, 0, 1, 2))
    out['driving_mesh'] = {'mesh': mesh_array, 'normed_mesh': normed_mesh_array, 'R': R_array, 't': t_array, 'c': c_array, 'z': z_array, 'driving_mesh': _mesh_array, 'driving_normed_mesh': _normed_mesh_array, 'driving_R': _R_array, 'driving_t': _t_array, 'driving_c': _c_array}
    out['driving_name'] = video_name
    out['source_name'] = video_name
    return out

def make_animation(source_video, driving_video, source_mesh, driving_mesh, driving_mesh_img, generator, cpu=False):
    visualizer = Visualizer(kp_size=4)
    num_frames = driving_video.shape[1]
    fps = 25
    times = np.linspace(0, num_frames / fps, num_frames)
    with torch.no_grad():
        source = torch.tensor(np.array(source_video)[np.newaxis].astype(np.float32))
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32))
        driving_mesh_img = torch.tensor(np.array(driving_mesh_img)[np.newaxis].astype(np.float32))

        searched_mesh = []
        normed_mesh = []
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_mesh_frame = driving_mesh_img[:, :, frame_idx]
            source_frame = source[:, :, frame_idx]

            kp_driving = preprocess_mesh(driving_mesh, frame_idx)
            kp_source = preprocess_mesh(source_mesh, frame_idx)

            if not cpu:
                driving_frame = driving_frame.cuda()
                source_frame = source_frame.cuda()
                # kp_driving['value'] = kp_driving['value'].cuda()
                # kp_source['value'] = kp_source['value'].cuda()

            out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving, driving_mesh_image=driving_mesh_frame, driving_image=driving_frame)

            driving_mesh_pos = kp_driving['mesh'].cuda()[:, :, None, :2] # B x K x 1 x 2
            driving_mesh_normalized_pos = kp_driving['normed_mesh'].cuda()[:, :]
            # motion = kp_driving['mesh'].cuda()
            motion = kp_driving['driving_normed_mesh'].cuda()
            motion[:, :, :2] = F.grid_sample(out['deformation'].permute(0, 3, 1, 2), driving_mesh_pos).squeeze(3).permute(0, 2, 1)   # B x K x 2
            motion[:, :, 2] = driving_mesh_normalized_pos[:, :, 2]


            # motion[LIP_IDX] = kp_driving['normed_lip']
            filename = '{:05d}.pt'.format(frame_idx + 1)

            R = kp_driving['driving_R'][0].cuda()
            RT = R.transpose(0, 1)
            t = kp_driving['driving_t'][0].cuda()
            c = kp_driving['driving_c'][0].cuda()

            t -= torch.matmul(R, (c * torch.ones_like(t)))
            c /= 128
            R = R

            # normalized_base = 128 * (kp_driving['normed_mesh'][0] + 1)
            # base = 128 * (kp_driving['mesh'][0] + 1)
            geometry = 128 * (motion.view(-1, 3) + 1)

            normalised_geometry = geometry.clone().detach().cpu()
            # normalised_geometry = mix_mesh_tensor(normalised_geometry, normalized_base.cpu())
            normalised_landmark_dict = mesh_tensor_to_landmarkdict(normalised_geometry)
            
            geometry = (torch.matmul(RT, (geometry.transpose(0, 1) - t)) / c).transpose(0, 1).cpu().detach()
            
            # mix with original geometry
            _geometry = kp_driving['driving_mesh'][0].cpu()
            _geometry[:] = geometry[:]
            geometry = _geometry

            geometry = geometry.numpy()
            x = geometry[:, 0]
            y = geometry[:, 1]
            z = geometry[:, 2]

            if frame_idx == 0:
                x_filter = OneEuroFilter(times[0], x[0], min_cutoff=MIN_CUTOFF, beta=BETA)
                y_filter = OneEuroFilter(times[0], y[0], min_cutoff=MIN_CUTOFF, beta=BETA)
                z_filter = OneEuroFilter(times[0], z[0], min_cutoff=MIN_CUTOFF, beta=BETA)
            else:
                x = x_filter(times[frame_idx], x)
                y = y_filter(times[frame_idx], y)
                z = z_filter(times[frame_idx], z)

            geometry = torch.from_numpy(np.stack([x, y, z], axis=1))

            # geometry = mix_mesh_tensor(geometry, base.cpu())
            landmark_dict = mesh_tensor_to_landmarkdict(geometry)
            landmark_dict.update({'R': R.cpu().numpy(), 't': t.cpu().numpy(), 'c': c.cpu().numpy()})
            torch.save(normalised_landmark_dict, os.path.join(opt.data_dir,'mesh_dict_searched_normalized',filename))
            torch.save(landmark_dict, os.path.join(opt.data_dir, 'mesh_dict_searched', filename))

            driving_mesh_frame = driving_mesh_frame[0].permute(1, 2, 0).detach().cpu().numpy()
            deformation = out['mask'][0].permute(1, 2, 0).detach().cpu().numpy()
            seg_img = visualizer.visualize_segment(driving_mesh_frame, deformation)
            imageio.imsave(os.path.join(opt.data_dir, 'seg_image', '{:05d}.png'.format(frame_idx + 1)), seg_img)


def save_searched_mesh(searched_mesh_batch, save_dir):
    # searched_mesh_batch: L x N * 3
    shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for i in tqdm(range(len(searched_mesh_batch))):
        mesh = searched_mesh_batch[i].view(-1, 3)   # N x 3
        mesh_dict = mesh_tensor_to_landmarkdict(mesh)
        torch.save(mesh_dict, os.path.join(save_dir, '{:05d}.pt'.format(i + 1)))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/kkj-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='log/sonny/last.tar', help="path to checkpoint to restore")
    parser.add_argument("--data_dir", default='data/sonny/test.mp4', help="video directory")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument('--device_id', type=str, default='1')
 

    opt = parser.parse_args()

    fps = 25

    generator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    dataset = get_dataset(opt.data_dir)

    init_dir(os.path.join(opt.data_dir, 'mesh_dict_searched'))
    init_dir(os.path.join(opt.data_dir, 'mesh_dict_searched_normalized'))
    init_dir(os.path.join(opt.data_dir, 'seg_image'))

    make_animation(dataset['video'], dataset['video'], dataset['mesh'], dataset['driving_mesh'], dataset['driving_mesh_img'], generator, cpu=opt.cpu)

    image_rows = image_cols = 256
    draw_mesh_images(os.path.join(opt.data_dir, 'mesh_dict_searched_normalized'), os.path.join(opt.data_dir, 'mesh_image_searched_normalized'), image_rows, image_cols)
    draw_mesh_images(os.path.join(opt.data_dir, 'mesh_dict_searched'), os.path.join(opt.data_dir, 'mesh_image_searched'), image_rows, image_cols)
    interpolate_zs(os.path.join(opt.data_dir, 'mesh_dict_searched'), os.path.join(opt.data_dir, 'z_searched'), image_rows, image_cols)
    interpolate_zs(os.path.join(opt.data_dir, 'mesh_dict_searched_normalized'), os.path.join(opt.data_dir, 'z_searched_normalized'), image_rows, image_cols)

