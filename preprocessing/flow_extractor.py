import argparse
import gc
import glob
import json
import multiprocessing
import os
import subprocess
from collections import OrderedDict

import flowiz as fz
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

from tools import *


def get_list_of_face_dirs(dir_path, dir_pattern='_faces'):
    """Get list of face directories to process"""
    list_of_files = os.listdir(dir_path)
    all_input_dirs = list()
    all_output_dirs = list()
    for entry in sorted(list_of_files):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):
            if entry.endswith(dir_pattern):
                all_input_dirs.append(full_path)
                all_output_dirs.append(get_path_same_dir(full_path, entry.split(dir_pattern)[0] + '_flows'))
            else:
                input_dirs, output_dirs = get_list_of_face_dirs(full_path)
                all_input_dirs += input_dirs
                all_output_dirs += output_dirs
    return all_input_dirs, all_output_dirs


def extract_flow(in_dir, out_dir=None):
    if out_dir is None:
        out_dir = get_path_same_dir(in_dir, os.path.basename(in_dir).split('_faces')[0] + '_flows')
    # print(in_dir, out_dir)
    cmd = ['python3', 'flownet2-pytorch/avspeech_extract_flow.py', '--number_gpus=1', '--skip_training', '--skip_validation', '--inference',\
        '--model=FlowNet2', '--resume=checkpoints/FlowNet2_checkpoint.pth.tar', '--save_flow', '--inference_dataset=ImagesFromFolder',\
            '--inference_dataset_iext=jpg', '--inference_dataset_root={}'.format(in_dir), '--save={}'.format(out_dir)]
    # print(' '.join(cmd))
    complete = subprocess.call(cmd, shell=False)
    return out_dir, complete


def process_json_data(input_json_path, output_json_path=None):
    """Extract all optical flows from input json"""
    print('Processing \"{}\"...'.format(input_json_path))
    dataset_path = ''
    num_videos = 0
    all_info = []
    with open(input_json_path, 'r') as fp:
        json_obj = json.load(fp)
        dataset_path = json_obj['dataset_path']
        num_videos = json_obj['num_videos']
        all_info = json_obj['files']

    output_json_obj = OrderedDict()
    output_json_obj['dataset_path'] = dataset_path
    output_json_obj['num_videos'] = num_videos

    output_json_clips = []
    for clip in tqdm(all_info):
        in_dir = clip['frames_path']
        # print(in_dir)
        # out_dir = get_path_same_dir(in_dir, os.path.basename(in_dir).split('_faces')[0] + '_flows')
        # print(out_dir)
        out_dir, _ = extract_flow(in_dir)
        clip['flows_path'] = out_dir
        output_json_clips.append(clip)
    output_json_obj['files'] = output_json_clips

    # write file
    if output_json_path is None:
        output_json_path = get_path_same_dir(input_json_path, os.path.basename(input_json_path).split('.json')[0] + '_flows.json')
    print('Writing to "{}"...'.format(output_json_path))
    with open(output_json_path, 'wb') as f:
        json_str = json.dumps(output_json_obj, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())

    print('Done')


def convert_flow(in_dir):
    files = glob.glob(os.path.join(in_dir, '*.flo'))
    for f in tqdm(sorted(files)):
        # save numpy array
        out_np = get_path_same_dir(f, os.path.splitext(os.path.basename(f))[0] + '.npy')
        # print(out_np)
        uv = fz.convert_from_file(f, mode='UV')
        # print(uv.shape)
        # print(np.min(uv), np.max(uv))
        np.save(out_np, uv)
        # new_uv = np.load(out_np)
        # print(np.array_equal(uv, new_uv))

        # save rgb image
        out_rgb = get_path_same_dir(f, os.path.splitext(os.path.basename(f))[0] + '.png')
        # print(out_rgb)
        img = fz.convert_from_file(f, mode='RGB')
        Image.fromarray(img).save(out_rgb)


def process_all_flofiles(input_json_path):
    """Convert all flo files to rgb images and numpy arrays"""
    print('Processing \"{}\"...'.format(input_json_path))
    all_info = []
    with open(input_json_path, 'r') as fp:
        json_obj = json.load(fp)
        all_info = json_obj['files']

    for clip in tqdm(all_info):
        in_dir = clip['flows_path']
        # print(in_dir)
        convert_flow(in_dir)

    print('Done')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--root', type=str, required=True, help='Specify the root of dataset directory')
    # args = parser.parse_args()
    # face_dirs, out_dirs = get_list_of_face_dirs(args.root)

    # for i in zip(face_dirs, out_dirs):
    #     extract_flow(*i)
    #     exit()

    # Parallel(n_jobs=-1, backend="multiprocessing")(delayed(extract_flow)(*i) for i in zip(face_dirs, out_dirs))

    TED_TRAINING_FACES_JSON = os.path.join(DATA_ROOT, 'training_TED_faces.json')
    TED_TESTING_FACES_JSON = os.path.join(DATA_ROOT, 'testing_TED_faces.json')
    TED_TRAINING_FACES_FLOWS_JSON = os.path.join(DATA_ROOT, 'training_TED_faces_flows.json')
    TED_TESTING_FACES_FLOWS_JSON = os.path.join(DATA_ROOT, 'testing_TED_faces_flows.json')

    # process_json_data(TED_TESTING_FACES_JSON, TED_TESTING_FACES_FLOWS_JSON)
    # process_all_flofiles(TED_TESTING_FACES_FLOWS_JSON)
    # process_json_data(TED_TRAINING_FACES_JSON, TED_TRAINING_FACES_FLOWS_JSON)
    process_all_flofiles(TED_TRAINING_FACES_FLOWS_JSON)
