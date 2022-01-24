"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""

import os
import cv2
import argparse
from tqdm import tqdm
import face_recognition

import util
import argparse
import torch
import ffmpeg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def paste_image(data_dir, patch_dir):
    patch_list = util.get_file_list(os.path.join(data_dir, patch_dir))
    print('number of patches: {}'.format(len(patch_list)))
    for i in tqdm(range(len(patch_list))):
        patch_path = patch_list[i]
        key = os.path.basename(patch_path)[:-4]
        crop_region = torch.load(os.path.join(data_dir, 'crop_region', key + '.pt'))
        top, bottom, left, right = crop_region
        image_path = os.path.join(data_dir, 'full', key + '.png')
        image = cv2.imread(image_path)
        patch = cv2.imread(patch_path)
        dest_h, dest_w = bottom - top, right - left
        image[top:bottom, left:right] = cv2.resize(patch, (dest_w, dest_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(args.data_dir, 'pasted', key + '.png'), image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default='data/sonny/test.mp4')
    parser.add_argument('--patch_dir', type=str, default='img')
    parser.add_argument('--name', type=str, default='pasted')
    parser.add_argument('--resize', type=int, default=None)


    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_dir, 'pasted'), exist_ok=True)

    paste_image(args.data_dir, args.patch_dir)
    if args.resize is None:
        ffmpeg.output(ffmpeg.input(os.path.join(args.data_dir, 'pasted', '%05d.png')), ffmpeg.input(os.path.join(args.data_dir, 'audio.wav')), os.path.join(args.data_dir, '{}.mp4'.format(args.name))).overwrite_output().run()
    else:
        ffmpeg.output(ffmpeg.input(os.path.join(args.data_dir, 'pasted', '%05d.png')).filter('scale', args.resize, args.resize), ffmpeg.input(os.path.join(args.data_dir, 'audio.wav')), os.path.join(args.data_dir, '{}.mp4'.format(args.name))).overwrite_output().run()
