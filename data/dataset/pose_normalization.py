"""
Crop upper boddy in every video frame, square bounding box is averaged among all frames and fixed.
"""
import sys

import os
import cv2
import argparse
import math
from tqdm import tqdm
import torch
import util
from util import get_file_list, landmark_to_dict, interpolate_zs, interpolate_z, normalize_meshes, normalize_mesh, landmarkdict_to_mesh_tensor, draw_mesh_images
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
from multiprocessing import Pool
import shutil

def get_reference_dict(ref_path):
    image = cv2.imread(ref_path)
    image_rows, image_cols, _ = image.shape

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        reference_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
        reference_dict = normalized_to_pixel_coordinates(reference_dict, image_cols, image_rows)
        target_dict = reference_dict.copy()
        R, t, c = util.Umeyama_algorithm(reference_dict, target_dict)
        target_dict['R'] = R
        target_dict['t'] = t
        target_dict['c'] = c
    return target_dict

def draw_landmark(results, image, save_path):
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=results.multi_face_landmarks[0],
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())

    cv2.imwrite(save_path, image)

def draw_mesh_image(mesh_dict, save_path, image_rows, image_cols):
    connections = mp_face_mesh.FACEMESH_TESSELATION
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)

    idx_to_coordinates = {}
    for idx, coord in mesh_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        x_px = min(math.floor(coord[0]), image_cols - 1)
        y_px = min(math.floor(coord[1]), image_rows - 1)
        landmark_px = (x_px, y_px)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    white_image = np.zeros([image_rows, image_cols, 3], dtype=np.uint8)
    white_image[:] = 0
    for edge_index, connection in enumerate(connections):
        start_idx = connection[0]
        end_idx = connection[1]
        color = edge2color(edge_index)
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(white_image, 
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], 
                color,
                drawing_spec.thickness
            )
    cv2.imwrite(save_path, white_image)

def normalized_to_pixel_coordinates(landmark_dict, image_width, image_height):
    def is_valid_normalized_value(value):
        return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    landmark_pixel_coord_dict = {}

    for idx, coord in landmark_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue

        if not (is_valid_normalized_value(coord[0]) and
                is_valid_normalized_value(coord[1])):

            return None
        x_px = coord[0] * image_width
        y_px = coord[1] * image_height
        z_px = coord[2] * image_width
        landmark_pixel_coord_dict[idx] = [x_px, y_px, z_px]
    return landmark_pixel_coord_dict

def edge2color(edge_index):
    total = len(mp_face_mesh.FACEMESH_TESSELATION)
    id = (edge_index + 1) / total
    c = 127 + int(id * 128)
    return (c, c, c)

def draw_pose_normalized_mesh(target_dict, image, save_path):
    connections = mp_face_mesh.FACEMESH_TESSELATION
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)

    image_rows, image_cols, _ = image.shape
    R = target_dict['R']
    t = target_dict['t']
    c = target_dict['c']

    idx_to_coordinates = {}
    for idx, coord in target_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        tgt = np.array(coord).reshape(3, 1)
        norm_tgt = (c * np.matmul(R, tgt) + t).squeeze()
        x_px = min(math.floor(norm_tgt[0]), image_cols - 1)
        y_px = min(math.floor(norm_tgt[1]), image_rows - 1)
        landmark_px = (x_px, y_px)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    
    white_image = np.zeros([image_rows, image_cols, 3], dtype=np.uint8)
    white_image[:] = 255
    for edge_index, connection in enumerat(connections):
        start_idx = connection[0]
        end_idx = connection[1]
        color = edge2color(edge_index)
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            cv2.line(white_image, 
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx], 
                color,
                drawing_spec.thickness
            )
    cv2.imwrite(save_path, white_image)


def draw_3d_mesh(target_dict, save_path, elevation=10, azimuth=10):
    connections = mp_face_mesh.FACEMESH_TESSELATION
    drawing_spec = mp_drawing.DrawingSpec(color= mp_drawing.BLACK_COLOR, thickness=1, circle_radius=1)

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)
    plotted_landmarks = {}

    for idx, coord in target_dict.items():
        if (idx == 'R') or (idx == 't') or (idx == 'c'):
            continue
        plotted_landmarks[idx] = (-coord[2], coord[0], -coord[1])

    for edge_index, connection in enumerate(connections):
        start_idx = connection[0]
        end_idx = connection[1]
        color = edge2color(edge_index)
        if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
            landmark_pair = [plotted_landmarks[start_idx], plotted_landmarks[end_idx]]
            ax.plot3D(
                xs=[landmark_pair[0][0], landmark_pair[1][0]],
                ys=[landmark_pair[0][1], landmark_pair[1][1]],
                zs=[landmark_pair[0][2], landmark_pair[1][2]],
                color=color,
                linewidth=1)
    plt.savefig(save_path)

def multiProcess(im, data_dir, reference_dict):
    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        image = cv2.imread(im)
        annotated_image = image.copy()
        image_rows, image_cols, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        target_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
        target_dict = normalized_to_pixel_coordinates(target_dict, image_cols, image_rows)
        R, t, c = util.Umeyama_algorithm(reference_dict, target_dict)
        target_dict['R'] = R
        target_dict['t'] = t
        target_dict['c'] = c
        torch.save(target_dict, os.path.join(data_dir, 'mesh_dict', os.path.basename(im))[:-4]+'.pt')

        if args.draw_mesh:
            img_save_path = os.path.join(data_dir, 'mesh_image', os.path.basename(im)[:-4] + '.png')
            draw_landmark(results, annotated_image, img_save_path)
            
        if args.draw_norm_mesh:
            img_save_path = os.path.join(data_dir, 'mesh_norm_image', os.path.basename(im)[:-4] + '.png')
            draw_pose_normalized_mesh(target_dict, annotated_image, img_save_path)

        if args.draw_norm_3d_mesh:
            img_save_path = os.path.join(data_dir, 'mesh_norm_3d_image', os.path.basename(im)[:-4] + '.png')
            draw_3d_mesh(target_dict, img_save_path, elevation=10, azimuth=10)

def pose_normalization(args):
    data_dir = args.data_dir
    image_list = get_file_list(os.path.join(data_dir, 'img'))
    reference_dict = get_reference_dict(args.ref_path)
    torch.save(reference_dict, os.path.join(data_dir, 'mesh_dict_reference.pt'))
    torch.save(normalize_mesh(reference_dict), os.path.join(data_dir, 'mesh_dict_reference_normalized.pt'))
    torch.save(interpolate_z(landmarkdict_to_mesh_tensor(normalize_mesh(reference_dict))), os.path.join(data_dir, 'z_reference_normalized.pt'))

    data_dirs = []
    reference_dicts = []

    for i in range(len(image_list)):
        data_dirs.append(data_dir)
        reference_dicts.append(reference_dict)

    num_errors = 0
    num_frames = len(image_list)

    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        for i in tqdm(range(len(image_list))):
            # print(f'image name: {image_list[i]}')
            image = cv2.imread(image_list[i])
            annotated_image = image.copy()
            image_rows, image_cols, _ = image.shape
            try:
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                target_dict = landmark_to_dict(results.multi_face_landmarks[0].landmark)
                target_dict = normalized_to_pixel_coordinates(target_dict, image_cols, image_rows)
                R, t, c = util.Umeyama_algorithm(reference_dict, target_dict)
                target_dict['R'] = R
                target_dict['t'] = t
                target_dict['c'] = c
            except:
                num_errors += 1

            torch.save(target_dict, os.path.join(data_dir, 'mesh_dict', os.path.basename(image_list[i]))[:-4]+'.pt')

    return num_frames, num_errors

def init_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def create_dirs(opt):
    init_dir(os.path.join(args.data_dir, 'mesh_dict'))
    if opt.draw_mesh:
        init_dir(os.path.join(args.data_dir, 'mesh_image'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--ref_path', type=str, default=None)
    parser.add_argument('--draw_mesh', action='store_true')
    parser.add_argument('--interpolate_z', action='store_true')

    args = parser.parse_args()

    data_dir = args.data_dir
    if args.ref_path is None:
        args.ref_path = os.path.join(data_dir, 'frame_reference.png')
    create_dirs(args)
    num_frames, num_errors = pose_normalization(args)
    normalize_meshes(os.path.join(data_dir, 'mesh_dict'), os.path.join(data_dir, 'mesh_dict_normalized'))

    if args.draw_mesh:
        draw_mesh_images(os.path.join(data_dir, 'mesh_dict'), os.path.join(data_dir, 'mesh_image'), 256, 256)

    if args.interpolate_z:
        interpolate_zs(os.path.join(data_dir, 'mesh_dict'), os.path.join(data_dir, 'z'), 256, 256)
        interpolate_zs(os.path.join(data_dir, 'mesh_dict_normalized'), os.path.join(data_dir, 'z_normalized'), 256, 256)

    print('pose normalization finished - total frames: {}, error frames: {}'.format(num_frames, num_errors))