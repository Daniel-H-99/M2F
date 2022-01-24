"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""

import os
import cv2
import argparse
from tqdm import tqdm
import face_recognition
import torch
import util
import numpy as np
import face_alignment

def calc_bbox(image_list, batch_size=5):
    """Batch infer of face location, batch_size should be factor of total frame number."""
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    top_best = 10000
    bottom_best = 0
    right_best = 0
    left_best = 10000

    for i in tqdm(range(len(image_list) // batch_size - batch_size)):
        image_batch = []

        for j in range(i * batch_size, (i + 1) * batch_size):
            image = face_recognition.load_image_file(image_list[j])
        
            # face_locations = face_recognition.batch_face_locations(image_batch, number_of_times_to_upsample=0, batch_size=batch_size)
            image = torch.tensor(image)

            preds = fa.face_detector.detect_from_image(torch.tensor(image))

            left, top, right, bottom, _ = preds[0]  # assuming only one face detected in the frame
            if top_best > top:
                top_best = top
            if bottom_best < bottom:
                bottom_best = bottom
            if right_best < right:
                right_best = right
            if left_best > left:
                left_best = left

    return top_best, right_best, bottom_best, left_best


def crop_image(data_dir, dest_size, crop_level, vertical_adjust):
    image_list = util.get_file_list(os.path.join(data_dir, 'full'))
    H, W, _ = face_recognition.load_image_file(image_list[0]).shape
    top, right, bottom, left = calc_bbox(image_list)
    height = bottom - top
    width = right - left

    crop_size = int(height * crop_level)

    horizontal_delta = (crop_size - width) // 2
    vertical_delta = (crop_size - height) // 2

    left = int(max(left - horizontal_delta, 0))
    right = int(min(right + horizontal_delta, W))

    top = int(max(top - int(vertical_delta * 0.5), 0))
    bottom =int(min(bottom + int(vertical_delta * 1.5), H))

    for i in tqdm(range(len(image_list))):
        image =cv2.imread(image_list[i])
        image = image[top:bottom, left:right]

        image = cv2.resize(image, (dest_size, dest_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(args.data_dir, 'crop', os.path.basename(image_list[i])), image)
        torch.save([top, bottom, left, right], os.path.join(data_dir, 'crop_region', os.path.basename(image_list[i]))[:-4]+'.pt')



def crop_per_image(data_dir, dest_size, crop_level, vertical_adjust):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

    image_list = util.get_file_list(os.path.join(data_dir, 'full'))
    batch_size = 1
    frames = []

    for i in tqdm(range(len(image_list))):
        frame = face_recognition.load_image_file(image_list[i])
        frames.append(frame)

    H, W, _ = frames[0].shape

    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    for idx in tqdm(range(len(batches))):
        fb = torch.tensor(batches[idx])
        pred = fa.face_detector.detect_from_image(fb[0])
        _left, _top, _right, _bottom, _ = pred[0]  # assuming only one face detected in the frame
        left = _left
        right = _right
        top = _top
        bottom = _bottom

        height = bottom - top
        width = right - left

        crop_size = int(height * crop_level)

        horizontal_delta = (crop_size - width) // 2
        vertical_delta = (crop_size - height) // 2

        left = int(max(left - horizontal_delta, 0))
        right = int(min(right + horizontal_delta, W))

        top = int(max(top - int(vertical_delta * 0.5), 0))
        bottom =int(min(bottom + int(vertical_delta * 1.5), H))

        crop_f = cv2.imread(image_list[idx * batch_size])
        crop_f = crop_f[top:bottom, left:right]
        crop_f = cv2.resize(crop_f, (dest_size, dest_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(data_dir, 'crop', os.path.basename(image_list[idx * batch_size])), crop_f)
        torch.save([top, bottom, left, right], os.path.join(data_dir, 'crop_region', os.path.basename(image_list[idx * batch_size]))[:-4]+'.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dest_size', type=int, default=256)
    parser.add_argument('--crop_level', type=float, default=1.4, help='Adjust crop image size.')
    parser.add_argument('--vertical_adjust', type=float, default=0.85, help='Adjust vertical location of portrait in image.')
    args = parser.parse_args()
    util.create_dir(os.path.join(args.data_dir, 'crop_region'))
    util.create_dir(os.path.join(args.data_dir, 'crop'))
    # crop_per_image(args.data_dir, dest_size=args.dest_size, crop_level=args.crop_level)
    crop_per_image(args.data_dir, dest_size=args.dest_size, crop_level=args.crop_level, vertical_adjust=args.vertical_adjust)
