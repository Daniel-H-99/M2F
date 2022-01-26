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
from sync_batchnorm import DataParallelWithCallback
from utils.util import mesh_tensor_to_landmarkdict, draw_mesh_images, interpolate_zs, mix_mesh_tensor, LIP_IDX, ROI_IDX, WIDE_MASK_IDX, get_lip_mask

from modules.generator import MeshOcclusionAwareGenerator
from scipy.spatial import ConvexHull
import os
import ffmpeg
import cv2


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def preprocess_mesh(m, frame_idx):
    roi = ROI_IDX
    res = m.copy()
    for key in res.keys():
        res[key] = torch.tensor(res[key][frame_idx])[None].float().cuda()
    res['value'] = res['normed_mesh'][:, roi, :2]
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
    roi_list = [0, 267, 13, 14, 269, 270, 17, 146, 402, 405, 409, 415, 37, 39, 40, 178, 181, 310, 311, 312, 185, 314, 317, 61, 191, 318, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375]
    video_name = os.path.basename(path)
    frames = sorted(os.listdir(os.path.join(path, 'img')))
    num_frames = len(frames) if opt.use_raw else len(os.listdir(os.path.join(path, 'mesh_dict_searched')))
    print(f'number of frames: {num_frames}')
    frames = frames[:num_frames]
    frame_idx = range(num_frames)

    reference_frame_path = os.path.join(path, 'frame_reference.png')
    reference_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference.pt'))
    reference_normed_mesh_dict = torch.load(os.path.join(path, 'mesh_dict_reference_normalized.pt'))
    reference_frame = img_as_float32(io.imread(reference_frame_path))
    reference_mesh = np.array(list(reference_mesh_dict.values())[:478])
    reference_normed_mesh = np.array(list(reference_normed_mesh_dict.values())[:478])
    reference_R = np.array(reference_mesh_dict['R'])
    reference_t = np.array(reference_mesh_dict['t'])
    reference_c = np.array(reference_mesh_dict['c'])
    reference_normed_z = torch.load(os.path.join(path, 'z_reference_normalized.pt'))
    video_array = [reference_frame for idx in frame_idx]
    mesh_array = [reference_mesh for idx in frame_idx]
    normed_mesh_array = [reference_normed_mesh for idx in frame_idx]
    z_array = [reference_normed_z for idx in frame_idx]
    R_array = [reference_R for idx in frame_idx]
    t_array = [reference_t for idx in frame_idx]
    c_array = [reference_c for idx in frame_idx]

    if opt.use_raw:
        mesh_dict = 'mesh_dict'
        normed_mesh_dict = 'mesh_dict_normalized'
        driving_mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_normed_mesh_array = [np.array(list(torch.load(os.path.join(path, normed_mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image', frames[idx]))) for idx in frame_idx]
        driving_video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[idx]))) for idx in frame_idx]
        driving_z_array = [torch.load(os.path.join(path, 'z', frames[idx].replace('.png', '.pt'))) for idx in frame_idx]
        driving_R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
        driving_t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
        driving_c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]
        lip_mask_array = [get_lip_mask(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))), (256, 256, 3)) for idx in frame_idx]

    else:
        mesh_dict = 'mesh_dict_searched'
        normed_mesh_dict = 'mesh_dict_searched_normalized'
        driving_mesh_array = [np.array(list(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_normed_mesh_array = [np.array(list(torch.load(os.path.join(path, normed_mesh_dict, frames[idx].replace('.png', '.pt'))).values())[:478]) for idx in frame_idx]
        driving_mesh_img_array = [img_as_float32(io.imread(os.path.join(path, 'mesh_image_searched', frames[idx]))) for idx in frame_idx]
        driving_video_array = [img_as_float32(io.imread(os.path.join(path, 'img', frames[idx]))) for idx in frame_idx]
        driving_z_array = [torch.load(os.path.join(path, 'z_searched', frames[idx].replace('.png', '.pt'))) for idx in frame_idx]
        driving_R_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['R']) for idx in frame_idx]
        driving_t_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['t']) for idx in frame_idx]
        driving_c_array = [np.array(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt')))['c']) for idx in frame_idx]
        lip_mask_array = [get_lip_mask(torch.load(os.path.join(path, mesh_dict, frames[idx].replace('.png', '.pt'))), (256, 256, 3)) for idx in frame_idx]
        
    video_array = np.array(video_array, dtype='float32')
    mesh_array = np.array(mesh_array, dtype='float32') / 128 - 1
    normed_mesh_array = np.array(normed_mesh_array, dtype='float32') / 128 - 1
    R_array = np.array(R_array, dtype='float32')
    c_array = np.array(c_array, dtype='float32') * 128
    t_array = np.array(t_array, dtype='float32')
    t_array = t_array + np.matmul(R_array, (c_array[:, None, None] * np.ones_like(t_array)))
    z_array = torch.stack(z_array, dim=0).float() / 128 - 1
    out = {}

    driving_video_array = np.array(driving_video_array, dtype='float32')
    driving_mesh_img_array = np.array(driving_mesh_img_array, dtype='float32')
    # driving_mesh_img_array = np.array(driving_mesh_img_array, dtype='float32') * np.array(lip_mask_array, dtype='float32')
    driving_mesh_array = np.array(driving_mesh_array, dtype='float32') / 128 - 1
    driving_normed_mesh_array = np.array(driving_normed_mesh_array, dtype='float32') / 128 - 1
    driving_z_array = torch.stack(driving_z_array, dim=0).float() / 128 - 1
    driving_R_array = np.array(driving_R_array, dtype='float32')
    driving_c_array = np.array(driving_c_array, dtype='float32') * 128
    driving_t_array = np.array(driving_t_array, dtype='float32')
    driving_t_array = driving_t_array + np.matmul(driving_R_array, (driving_c_array[:, None, None] * np.ones_like(driving_t_array)))

    video = video_array
    out['video'] = video.transpose((3, 0, 1, 2))
    out['mesh'] = {'mesh': mesh_array, 'normed_mesh': normed_mesh_array, 'R': R_array, 't': t_array, 'c': c_array, 'normed_z': z_array}
    out['driving_video'] = driving_video_array.transpose((3, 0, 1, 2))
    out['driving_mesh_img'] = driving_mesh_img_array.transpose((3, 0, 1, 2))
    out['driving_mesh'] = {'mesh': driving_mesh_array, 'normed_mesh': driving_normed_mesh_array, 'R': driving_R_array, 't': driving_t_array, 'c': driving_c_array, 'z': driving_z_array}
    out['driving_name'] = video_name
    out['source_name'] = video_name
    return out

def make_animation(source_video, driving_video, source_mesh, driving_mesh, driving_mesh_img, generator, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(np.array(source_video)[np.newaxis].astype(np.float32))
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32))
        driving_mesh_img = torch.tensor(np.array(driving_mesh_img)[np.newaxis].astype(np.float32))

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            driving_mesh_frame = driving_mesh_img[:, :, frame_idx]
            source_frame = source[:, :, frame_idx]

            kp_driving = preprocess_mesh(driving_mesh, frame_idx)
            kp_source = preprocess_mesh(source_mesh, frame_idx)

            if not cpu:
                driving_frame = driving_frame.cuda()
                source_frame = source_frame.cuda()

            out = generator(source_frame, kp_source=kp_source, kp_driving=kp_driving, driving_mesh_image=driving_mesh_frame, driving_image=driving_frame)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    return predictions

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/kkj-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='log/sonny/last.tar', help="path to checkpoint to restore")
    parser.add_argument("--data_dir", default='data/sonny/test.mp4', help="video directory")
    parser.add_argument("--result_video", default='recon.mp4', help="path to output")
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--use_raw", action="store_true", help="use raw dataset")
    parser.add_argument('--device_id', type=str, default='1')

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_id

    fps = 25

    generator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    generator.module.dense_motion_network.prior_from_audio = False
    dataset = get_dataset(opt.data_dir)
    predictions = make_animation(dataset['video'], dataset['driving_video'], dataset['mesh'], dataset['driving_mesh'], dataset['driving_mesh_img'], generator, cpu=opt.cpu)
    os.makedirs(os.path.join(opt.data_dir, 'demo_img'), exist_ok=True)
    for i, pred in tqdm(enumerate(predictions)):
        cv2.imwrite(os.path.join(opt.data_dir, 'demo_img', '{:05d}.png'.format(i + 1)), img_as_ubyte(pred)[:, :, [2, 1, 0]])
    imageio.mimsave(os.path.join(opt.data_dir, 'pre_' + opt.result_video), [img_as_ubyte(frame) for frame in predictions], fps=fps)
    if opt.use_raw:
        audio_name = '_audio.wav'
    else:
        audio_name = 'audio.wav'
    ffmpeg.output(ffmpeg.input(os.path.join(opt.data_dir, 'pre_' + opt.result_video)), ffmpeg.input(os.path.join(opt.data_dir, audio_name)), os.path.join(opt.data_dir, opt.result_video)).overwrite_output().run()