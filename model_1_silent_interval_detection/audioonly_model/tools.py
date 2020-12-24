import gc
import glob
import json
import math
import os
import pprint
import random
import re
import shutil
import subprocess
import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import groupby

import librosa
import librosa.display
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from backports import tempfile
from matplotlib.animation import FFMpegWriter
from PIL import Image
from torchvision import transforms as trn
import imageio

from transform import *
# from common import PHASE_TRAINING, PHASE_TESTING, PHASE_PREDICTION
from utils import get_parent_dir
import time
# import warnings
# warnings.filterwarnings("error")


JSON_DUMP_PARAMS = dict(indent=4, sort_keys=False, ensure_ascii=False, separators=(',', ':'))
EXTENSION = '.mp4'
TF = trn.Compose([
        trn.Resize((256, 256)),
        trn.ToTensor(),
        # trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

TRAIN_DATA_RATIO = 0.8
PRED_DATA_RATIO = 0.5

PLOT_W = 50
PLOT_H = 10
PLOT_DPI = 100
PLOT_CHUNKSIZE = 1e10
Y_LIM_T = 1.2
Y_LIM_B = -1.2

BITSTREAM_JSON_LABEL = 'bit_stream'

MAX_NUM_OF_0_IN_1_DATA = 4


def sigmoid(x):
    """Sigmoid function with numpy array"""
    return 1 / (1 + np.exp(-x))


def gauss_weights(n, sigma=None):
    """Generate 1D Gaussian kernel as weights"""
    if sigma is None:
        sigma = math.floor(math.sqrt((n**2-1)/12))
    if n % 2 == 0:
        r = np.linspace(-int(n/2)+0.5, int(n/2)-0.5, n)
    else:
        r = np.linspace(-int(n/2), int(n/2), n)
    weights = [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]
    return np.array([x / sum(weights) for x in weights])


def uniform_weights(n):
    """Generate 1D Uniform kernel as weights"""
    return np.full(n, float(1.0/n), np.float)


def audio_normalize(snd):
    """Normalize librosa audio array"""
    max_abs = max(abs(snd))
    if max_abs > 1:
        mult_var = 1.0 / max_abs
        return snd * mult_var
    else:
        return snd


def show_metrics(y_true, y_score, suppress_stdout=True):
    """Calculate different evaluation metrics"""
    y_true = np.int_(y_true)
    y_score = np.int_(y_score)

    # check number of 0/1 samples
    # class_sample_count = {0:len(np.where(y_true == 0)), 1:len(np.where(y_true == 1))}
    class_sample_count = dict([(int(t), len(np.where(y_true == t)[0]))\
        for t in np.unique(y_true)])
    if len(class_sample_count.keys()) == 1:
        class_sample_count[1-list(class_sample_count.keys())[0]] = 0
    # print(class_sample_count)

    # Calculate baseline and real accuracy
    base = sum(1 for i in y_true if i == 1.0) / len(y_true)
    accuracy = sum(1 for i, j in zip(y_true, y_score) if i == j) / len(y_true)

    # Flip arrays
    # 0 is positive; 1 is negative in this case
    y_true = 1 - y_true
    y_score = 1 - y_score

    # True positive
    tp = np.sum(y_true * y_score)
    # False positive
    fp = np.sum((y_true == 0) * y_score)
    # True negative
    tn = np.sum((y_true == 0) * (y_score == 0))
    # False negative
    fn = np.sum(y_true * (y_score == 0))

    # True positive rate (sensitivity or recall)
    try:
        tpr = tp / (tp + fn)
    except:
        tpr = 0
    # False positive rate (fall-out)
    try:
        fpr = fp / (fp + tn)
    except:
        fpr = 0
    # Precision
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    try:
        f1 = 2*tp / (2*tp + fp + fn)
    except:
        f1 = 0
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    try:
        mcc = 0 if np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) == 0 else\
            (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    except:
        mcc = 0

    if not suppress_stdout:
        print("Number of samples: ", len(y_true))
        print("Number of silent samples: ", class_sample_count[0])
        print("Number of non-silent samples: ", class_sample_count[1])

        print("Base: ", base)
        print("Accuracy: ", accuracy)

        print("True positive: ", tp)
        print("False positive: ", fp)
        print("True negative: ", tn)
        print("False negative: ", fn)

        print("True positive rate (recall): ", tpr)
        print("False positive rate: ", fpr)
        print("Precision: ", precision)
        print("True negative rate: ", tnr)
        print("F1 (range: [0, 1]): ", f1)
        print("ROC-AUC (range: [0, 1]): ", auc)
        print("MCC (range: [-1, 1]): ", mcc)

    def nan_to_null(num):
        if np.isnan(num):
            return None
        return num

    # save stat results
    return OrderedDict([
        ('num_samples', len(y_true)),
        ('num_silent_samples', class_sample_count[0]),
        ('num_non_silent_samples', class_sample_count[1]),
        ('base', base),
        ('accuracy', accuracy),
        ('true_positive', nan_to_null(int(tp))),
        ('false_positive', nan_to_null(int(fp))),
        ('true_negative', nan_to_null(int(tn))),
        ('false_negative', nan_to_null(int(fn))),
        ('true_pos_rate(recall)', nan_to_null(float(tpr))),
        ('false_pos_rate', nan_to_null(float(fpr))),
        ('precision', nan_to_null(float(precision))),
        ('true_neg_rate', nan_to_null(float(tnr))),
        ('f1', nan_to_null(float(f1))),
        ('roc_auc', nan_to_null(float(auc))),
        ('mcc', nan_to_null(float(mcc)))
    ])


def load_image(path):
    """Load an image"""
    return TF(Image.open(path))


def load_image_from_arr(arr):
    """Load an image from an array"""
    return TF(Image.fromarray(arr))


def load_image_from_index(index, dir_path, randrot=0):
    """Load an image from index"""
    # print('loading {}...'.format(index))
    if randrot != 0:
        TF_ROT = trn.Compose([
            trn.Resize((256, 256)),
            trn.RandomRotation((randrot, randrot)),
            trn.ToTensor(),
        ])
        return TF_ROT(Image.open(\
            os.path.abspath('{}/{:07d}.jpg'.format(dir_path, index+1))))
    else:
        return TF(Image.open(\
            os.path.abspath('{}/{:07d}.jpg'.format(dir_path, index+1))))


def crop_face(img, x, y, flip=False):
    """Crop image using x and y coordinates"""
    magic_hw_ration = 1.75
    w, h = img.size
    dists_center = {
        'l': w * x,
        'r': w * (1.0 - x),
        't': h * y,
        'b': h * (1.0 - y)
    }
    smallest = min(dists_center, key=dists_center.get)
    if smallest in ('t', 'b'):
        crop_tuple = (
            dists_center['l'] - dists_center[smallest],
            dists_center['t'] - dists_center[smallest],
            w - (dists_center['r'] - dists_center[smallest]),
            h - (dists_center['b'] - dists_center[smallest])
        )
    else:
        crop_tuple = (
            dists_center['l'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['l'] else 0,
            dists_center['t'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['t'] else 0,
            w - (dists_center['r'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['r'] else 0),
            h - (dists_center['b'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['b'] else 0)
        )
    img1 = img.crop(crop_tuple)
    if flip:
        img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
    return img1


def load_face_from_index(index, dir_path, x, y, flip=False):
    """Load an image from index, only face region"""
    # print('loading {}...{}'.format(dir_path, index))
    # print('face x:', x)
    # print('face y:', y)
    return TF(crop_face(\
        Image.open(os.path.abspath('{}/{:07d}.jpg'.format(dir_path, index+1))), x, y, flip))


def truncate(l):
    """Truncate the leading and trailing '2' characters"""
    indices = [len(list(g)) for k, g in groupby(l) if k == '2']
    # return l[indices[0]:-indices[1]]
    return (indices[0], -indices[1])


def filter_bitstream(bs, min_silent_interval):
    """Filter out silent intervals shorter than min_silent_interval"""
    gt_bs_fixed = np.array(list(bs))
    gt_bs_fixed_groups = [(k, len(list(g))) for k, g in groupby(gt_bs_fixed)]
    gt_bs_fixed_groups_with_idx = []
    g_idx = 0
    for g in gt_bs_fixed_groups:
        gt_bs_fixed_groups_with_idx.append((*g, g_idx))
        g_idx += g[1]
    overwritten_gt_bs = []
    for g in gt_bs_fixed_groups_with_idx:
        if g[0] == '0' and g[1] < min_silent_interval:
            overwritten_gt_bs += list(range(g[2], g[2]+g[1]))
    if overwritten_gt_bs:
        overwritten_gt_bs = np.array(overwritten_gt_bs)
        gt_bs_fixed[overwritten_gt_bs] = '1'
    gt_bs_fixed = ''.join(gt_bs_fixed)
    return gt_bs_fixed


def bit_stream_indices_list(files, clip_frames, silent_consecutive_frames, random_seed=None, pred=False):
    """Create a list of truncated bit streams indices"""
    random.seed(random_seed)
    # bitstream_json_label = 'bit_stream_relabeled2'

    lists = []
    for i, f in enumerate(files):
        # print(f['path'])

        try:
            idx1, idx2 = truncate(f[BITSTREAM_JSON_LABEL])
        except:
            idx1 = 0
            idx2 = len(f[BITSTREAM_JSON_LABEL])
        cur_clip_bs = f[BITSTREAM_JSON_LABEL][idx1:idx2]   # actual bits
        # cur_clip_bs = filter_bitstream(cur_clip_bs, silent_consecutive_frames)   # filtered bits

        if not pred:
            cur_clip_bs_idx = list(range(len(cur_clip_bs)))[idx1:idx2+1-clip_frames:clip_frames//2]  # index # overlap half of data

            for x in cur_clip_bs_idx:
                # get data bits
                data_bits_labels = cur_clip_bs[x:x+clip_frames]
                # construct choice (data) lists - list of tuples
                # item[0]: video clip index
                # item[1]: data first bit's index in video clip
                # item[2]: data bit stream
                # item[3]: audio_path
                # item[4]: framerate
                choice = (i, x, [int(x) for x in data_bits_labels], f['audio_path'], f['framerate'])
                lists.append(choice)
        else:
            choice = (i, idx1, [int(x) for x in cur_clip_bs], f['audio_path'], f['framerate'])
            lists.append(choice)

    return lists


def create_sample_list_from_indices(files, num_samples=None, clip_frames=1, silent_consecutive_frames=1, random_seed=None, pred=False):
    """Create a list of samples from indices"""
    all_choices = bit_stream_indices_list(files, clip_frames, silent_consecutive_frames, random_seed=random_seed, pred=pred)
    # each choice: tuple(file index, file's bitstream index, data bitstream, data center bit, bce weights)
    print('Total available samples: ', len(all_choices))

    if num_samples is None:
        return all_choices

    np.random.seed(random_seed)
    
    # get weights
    # all_data_labels = [x[3] for x in all_choices]
    # class_sample_count = np.array([len(np.where(all_data_labels == t)[0]) for t in np.unique(all_data_labels)])
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[int(t)] for t in all_data_labels])
    # samples_weight /= sum(samples_weight)

    # all_chosen_indices = sorted(np.random.choice(len(all_choices), num_samples, replace=False, p=samples_weight))
    all_chosen_indices = sorted(np.random.choice(len(all_choices), num_samples, replace=False))
    result = [all_choices[i] for i in all_chosen_indices]
    return result

"""
def get_centerbit(bit_labels, silent_consecutive_frames):
    # print("labels:", bit_labels)
    center_bit_idx = len(bit_labels) // 2
    # print("idx:", center_bit_idx)
    center_bit = bit_labels[center_bit_idx]
    # print("center bit:", center_bit)
    groups = [(k, len(list(g))) for k, g in groupby(bit_labels)]
    # print(groups)

    idx = -1
    for item in groups:
        idx += item[1]
        # print('current idx:', idx)
        if center_bit_idx <= idx:
            # print("In group:", item)
            if item[0] == '0':
                if item[1] < silent_consecutive_frames:
                    # print('current silent interval not long enough')
                    center_bit = '1'
            break

    # print("final center bit:", center_bit)
    return center_bit
"""

# experiment: look for the label of the CENTER 5 frames
def get_centerbit(bit_labels, silent_consecutive_frames):
    """Get the center bit from the bit stream"""
    # print("labels:", bit_labels)
    center_bit_idx = len(bit_labels) // 2
    min_bit_idx = center_bit_idx - silent_consecutive_frames // 2
    max_bit_idx = min_bit_idx + silent_consecutive_frames
    center_bit_labels = bit_labels[min_bit_idx:max_bit_idx]
    # print("center labels:", center_bit_labels)

    groups = [(k, len(list(g))) for k, g in groupby(center_bit_labels)]
    # print(groups)
    if len(groups) == 1 and groups[0][0] == '0':
        return '0'
    else:
        return '1'


def get_absolute_centerbit(bit_labels):
    """Get the absolute center bit from the bit stream"""
    center_bit_idx = len(bit_labels) // 2
    return bit_labels[center_bit_idx]


def get_bce_weights(bit_labels):
    """Calculate bce weights"""
    # weights = {'silent':0.0, 'nonsilent':0.0}
    weights = [0.0, 0.0]
    center_bit_idx = len(bit_labels) // 2
    center_bit = bit_labels[center_bit_idx]

    left = bit_labels[:center_bit_idx]
    l_count = 0
    for i in reversed(left):
        l_count += 1
        if i != center_bit:
            break
        if l_count == center_bit_idx and i == center_bit:
            l_count += 1

    right = bit_labels[center_bit_idx+1:]
    r_count = 0
    for i in right:
        r_count += 1
        if i != center_bit:
            break
        if r_count == center_bit_idx and i == center_bit:
            r_count += 1

    min_dist = min(l_count, r_count)
    ratio = min_dist / (center_bit_idx + 1)
    assert 0 <= ratio <= 1
    weights[int(center_bit)] = ratio
    return weights


def save_imgs_from_tensor(frames, output_dir, suppress_stdout=True):
    """Save frames from tensor to visualize input / result"""
    if not suppress_stdout:
        print('Saving input frames...')

    frames_copy = frames.clone()
    frames_copy = frames_copy.squeeze(0).permute(1, 2, 3, 0)
    frames_np = frames_copy.detach().cpu().numpy()
    if not suppress_stdout:
        print(frames_np.shape)

    frames_list = []
    for i in range(frames_np.shape[0]):
        if not suppress_stdout:
            print(i)
            print(frames_np[i].shape)
            print(frames_np[i])

        img = Image.fromarray((frames_np[i]*255).astype('uint8'))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        frame_path = os.path.join(output_dir, 'img_{:03}.png'.format(i))
        # frames_list.append(frame_path)
        frames_list.append(img)
        img.save(frame_path)

    if not suppress_stdout:
        print('Done')
    return frames_list


def random_crop_batch(frames, size=224):
    """
    :param frames: tensor (B, 3, 256, 256)
    :return:
    """
    # print('random_crop_batch')
    ori_size = frames.size(-1)
    start = random.randint(0, ori_size - size)
    fixed_start = ori_size - size
    # print('[{}:{}, {}:{}]'.format(fixed_start, fixed_start+size, start, start+size))
    # keep random crop include the bottom most part
    frames = frames[:, :, fixed_start:fixed_start+size, start:start+size]
    return frames


def center_crop_batch(frames, size=224):
    """
    :param frames: tensor (B, 3, 256, 256)
    :return:
    """
    # print('center_crop_batch')
    ori_size = frames.size(-1)
    start = (ori_size - size) // 2
    fixed_start = ori_size - size
    # print('[{}:{}, {}:{}]'.format(fixed_start, fixed_start+size, start, start+size))
    frames = frames[:, :, fixed_start:fixed_start+size, start:start+size]
    return frames


def random_lrflip_batch(frames):
    """
    :param frames: tensor (B, 3, 256, 256)
    :return:
    """
    if random.uniform(0, 1) < 0.5:
        return torch.flip(frames, [3])
    else:
        return frames


def create_gif1(filenames, gif_path, fps=15.0):
    """Create gif from list of files"""
    # print(filenames)
    # print(gif_path)

    os.system('convert -loop 0 {} {}'.format(' '.join(filenames), gif_path))


def create_gif2(filenames, gif_path, fps=15.0):
    """Create gif from list of files"""
    # print(filenames)
    # print(gif_path)

    images = []
    for filename in filenames:
        # images.append(Image.open(filename))
        images.append(imageio.imread(filename))
    # print(len(images))
    # print(images[0])
    imageio.mimsave(gif_path, images, format='GIF', fps=fps)


def create_gif3(images, gif_path, fps=15.0):
    """Create gif from list of files"""
    # print(filenames)
    # print(gif_path)

    images[0].save(gif_path, format='GIF', append_images=images[1:], save_all=True, duration=1000.0/fps, loop=0)


def weighted_binary_cross_entropy(output, target, weights=None, epsilon=1e-8):
    """Custom weighted binary cross entropy loss function"""
    output = torch.sigmoid(output)

    if weights is not None:
        assert weights.size()[0] == target.size()[0]
        assert output.size() == target.size()

        loss = weights[:, 1] ** 2 * (target * torch.log(output + epsilon)) + \
               weights[:, 0] ** 2 * ((1 - target) * torch.log(1 - output + epsilon))
    else:
        loss = target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon)

    return torch.neg(torch.mean(loss))


class WeightedBCE():
    """Custom weighted binary cross entropy loss function"""
    def __init__(self, class_weights=None, epsilon=1e-8, **kwargs):
        self.class_weights = class_weights
        self.epsilon = epsilon

    def __call__(self, output, target, **kwargs):
        output = torch.sigmoid(output)

        if self.class_weights is not None:
            assert self.class_weights.size()[0] == target.size()[0]
            assert output.size() == target.size()

            loss = self.class_weights[:, 1] ** 2 * (target * torch.log(output + self.epsilon)) + \
                   self.class_weights[:, 0] ** 2 * ((1 - target) * torch.log(1 - output + self.epsilon))
        else:
            loss = target * torch.log(output + self.epsilon) + (1 - target) * torch.log(1 - output + self.epsilon)

        loss = torch.neg(torch.mean(loss))

        return loss


def plot_wav(snd, sr, downsample=False, plot_path=None, suppress_stdout=False):
    """Draw waveform plot"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    fig = plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    fig.subplots_adjust(wspace=None, hspace=None)
    ax = fig.add_subplot(111)
    ax.set_ylim(Y_LIM_B, Y_LIM_T)

     # draw waveform
    if downsample:
        librosa.display.waveplot(snd, sr=sr, max_sr=100, ax=ax)
    else:
        librosa.display.waveplot(snd, sr=sr, ax=ax)
    # plt.show()
    # print(plt.gca().get_xlim(), plt.gca().get_ylim())

    # save plot
    if plot_path is not None:
        plt.savefig(plot_path)

    plt.close('all')

    if not suppress_stdout:
        print('Waveform plot complete.')


def plot_wav_bitstream_overlay(bit_stream, snd, sr, downsample=False, plot_path=None, suppress_stdout=False):
    """Draw waveform and bitstream plots overlay"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    fig = plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    fig.subplots_adjust(wspace=None, hspace=None)
    ax = fig.add_subplot(111)
    ax.set_ylim(Y_LIM_B, Y_LIM_T)

    # draw waveform
    if downsample:
        librosa.display.waveplot(snd, sr=sr, max_sr=100, ax=ax)
    else:
        librosa.display.waveplot(snd, sr=sr, ax=ax)
    # plt.show()
    # print(plt.gca().get_xlim(), plt.gca().get_ylim())

    # draw bitstream
    # Highlight silent intervals
    # from itertools import groupby
    fps_factor = float(len(bit_stream) / plt.gca().get_xlim()[1])
    start_idx = 0
    for item in ((k, len(list(g))) for k, g in groupby([int(s) for s in bit_stream])):
        if item[0] == 0:    # highlight silent intervals
            ax.axvspan(start_idx/fps_factor, (start_idx+item[1])/fps_factor, color='#FF7043', alpha=0.35)
        elif item[0] == 2:   # highlight excluded intervals
            ax.axvspan(start_idx/fps_factor, (start_idx+item[1])/fps_factor, color='#78909C', alpha=0.35)
        start_idx += item[1]

    # save plot
    if plot_path is not None:
        plt.savefig(plot_path)

    plt.close('all')

    if not suppress_stdout:
        print('Waveform bistream overlay plot complete.')


def plot_wav_floatstreams_overlay(float_streams, snd, sr, downsample=False, labels=None, plot_path=None, suppress_stdout=False):
    if not isinstance(float_streams, np.ndarray):
        raise Exception('Wrong input type')

    if float_streams.ndim != 2:
        raise Exception('Wrong input dims')

    """Draw waveform and bitstream plots overlay"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    fig = plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    fig.subplots_adjust(wspace=None, hspace=None)
    ax = fig.add_subplot(111)
    ax.set_ylim(Y_LIM_B, Y_LIM_T)

    # draw waveform
    if downsample:
        librosa.display.waveplot(snd, sr=sr, max_sr=100, ax=ax)
    else:
        librosa.display.waveplot(snd, sr=sr, ax=ax)
    # plt.show()
    # print(plt.gca().get_xlim(), plt.gca().get_ylim())

    # draw floatstreams
    if labels is not None:
        assert len(labels) == float_streams.shape[0]

    x = [i for i in range(len(float_streams[0]))]
    for i in range(float_streams.shape[0]):
        y = list(float_streams[i])
        plt.plot(x, y, linewidth=5, marker="o", label=labels[i])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize=27)
    plt.gca().set_axisbelow(True)
    plt.grid(True)
    # plt.tight_layout()

    minx = min(x)
    maxx = max(x)
    plt.xlim(left=minx, right=maxx)
    miny = np.amin(float_streams) * 1.1
    maxy = np.amax(float_streams) * 1.1
    maxy = max(abs(miny), abs(maxy))
    miny = maxy * -1
    plt.ylim(bottom=miny, top=maxy)

    # save plot
    if plot_path is not None:
        plt.savefig(plot_path)

    plt.close('all')

    if not suppress_stdout:
        print('Waveform bistream overlay plot complete.')


def plot_wav_bitstream_floatstreams_overlay(bit_stream, float_streams, snd, sr, downsample=False, labels=None, plot_path=None, suppress_stdout=False):
    if not isinstance(float_streams, np.ndarray):
        raise Exception('Wrong input type')

    if float_streams.ndim != 2:
        raise Exception('Wrong input dims')

    assert len(bit_stream) == len(float_streams[0])

    """Draw waveform and bitstream plots overlay"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    fig = plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    fig.subplots_adjust(wspace=None, hspace=None)
    ax = fig.add_subplot(111)
    ax.set_ylim(Y_LIM_B, Y_LIM_T)

    # draw waveform
    if downsample:
        librosa.display.waveplot(snd, sr=sr, max_sr=100, ax=ax)
    else:
        librosa.display.waveplot(snd, sr=sr, ax=ax)
    # plt.show()
    # print(plt.gca().get_xlim(), plt.gca().get_ylim())

    # draw bitstream
    # Highlight silent intervals
    # from itertools import groupby
    fps_factor = float(len(bit_stream) / plt.gca().get_xlim()[1])
    start_idx = 0
    for item in ((k, len(list(g))) for k, g in groupby([int(s) for s in bit_stream])):
        if item[0] == 0:    # highlight silent intervals
            ax.axvspan(start_idx/fps_factor, (start_idx+item[1])/fps_factor, color='#FF7043', alpha=0.35)
        elif item[0] == 2:   # highlight excluded intervals
            ax.axvspan(start_idx/fps_factor, (start_idx+item[1])/fps_factor, color='#78909C', alpha=0.35)
        start_idx += item[1]

    # draw floatstreams
    if labels is not None:
        assert len(labels) == float_streams.shape[0]

    x = [i for i in range(len(float_streams[0]))]
    for i in range(float_streams.shape[0]):
        y = list(float_streams[i])
        plt.plot(x, y, linewidth=5, marker="o", label=labels[i])

    # Put a legend below current axis
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5, fontsize=27)
    plt.gca().set_axisbelow(True)
    plt.grid(True)
    # plt.tight_layout()

    minx = min(x)
    maxx = max(x)
    plt.xlim(left=minx, right=maxx)
    miny = np.amin(float_streams) * 1.1
    maxy = np.amax(float_streams) * 1.1
    maxy = max(abs(miny), abs(maxy))
    miny = maxy * -1
    plt.ylim(bottom=miny, top=maxy)

    # save plot
    if plot_path is not None:
        plt.savefig(plot_path)

    plt.close('all')

    if not suppress_stdout:
        print('Waveform bistream overlay plot complete.')


def convert_bitstreammask_to_audiomask(ref_audio_signal, frames_to_audiosample_ratio, bitstream):
    mask = np.zeros_like(ref_audio_signal)
    for bit_idx, bit in enumerate(bitstream):
        # mask out non-silent intervals in ref_audio_signal
        # silent 1. non-silent 0
        if bit == 0:    # silent frame
            mask[int(bit_idx * frames_to_audiosample_ratio):int((bit_idx+1) * frames_to_audiosample_ratio - 1)] = 1
        elif bit == 1:  # non-silent frame
            mask[int(bit_idx * frames_to_audiosample_ratio):int((bit_idx+1) * frames_to_audiosample_ratio - 1)] = 0
        else:
            # mask[int(bit_idx * frames_to_audiosample_ratio):int((bit_idx+1) * frames_to_audiosample_ratio - 1)] = 0
            print('Invalid bit?')
            raise RuntimeError

    # check if mask has sporatic 0/1's
    mask_idx = 0
    for k, g in groupby(mask):
        g_len = len(list(g))
        if g_len < 5:
            mask[mask_idx:mask_idx+g_len] = 1 - k
        mask_idx += g_len
    # print(mask)
    return mask


### AUDIO PROCESSING ###
def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]


def power_of_signal(signal):
    return np.sum(np.abs(signal ** 2))


def add_signals(signal, noises, snr, norm=0.5):
    """
    add signal and noise at given SNR
    :param signal:
    :param noise:
    :param snr: measured in dB
    :param norm: normalize the resulting audio within range
    :return:
        mixed, clean, noises
    """
    if not isinstance(noises, list):
        noises = [noises]

    signal_power = power_of_signal(signal)
    pn = signal_power / np.power(10, snr / 10)
    new_noises = []
    ret_signal = np.copy(signal)

    for noise in noises:
        if signal_power == 0:
            new_noise = noise
            new_noises.append(new_noise)
            ret_signal += new_noise
        else:
            ratio = np.sqrt(power_of_signal(noise)) / np.sqrt(pn)
            if ratio == 0:
                new_noise = noise
            else:
                new_noise = noise / ratio
            new_noises.append(new_noise)
            ret_signal += new_noise

    if norm:
        scale = np.max(np.abs(ret_signal)) / norm
        if scale != 0:
            return ret_signal / scale, signal / scale, [x / scale for x in new_noises]
        else:
            return ret_signal, signal, new_noises
    else:
        return ret_signal, signal, new_noises


def add_noise_to_audio(audio, noise, snr, start_pos=None, norm=0.5):
    # randomly load noise and randomly select an interval
    if start_pos is None:
        if len(noise) - len(audio) >= 1:
            start = random.randint(0, len(noise) - len(audio))
        elif len(noise) - len(audio) == 0:
            start = 0
        else:
            print('len(noise):', len(noise))
            print('len(audio):', len(audio))
            raise ValueError
        noise_cropped = noise[start:start+len(audio)]
    else:
        noise_cropped = noise[start_pos:start_pos+len(audio)]
        # print(len(noise_cropped))

    if (len(noise_cropped) < len(audio)):
        noise_cropped = np.concatenate((noise_cropped, np.zeros(len(audio) - len(noise_cropped))))
    assert len(noise_cropped) == len(audio)

    # create noisy signal input
    mixed_signal, clean_signal, noise_signal = add_signals(audio, [noise_cropped], snr=snr, norm=norm)

    return mixed_signal, clean_signal, noise_signal


def convert_snr_to_suffix(snrs, snr_idx):
    result = ""
    if snr_idx is not None:
        try:
            result = '_snr' + str(snrs[snr_idx]).replace('.', '_')
        except:
            pass
    return result


def convert_snr_to_suffix2(snr):
    result = ""
    if snr is not None:
        try:
            snr = float(snr)
            snr = int(snr) if snr.is_integer() else float(snr)
            result = '_snr' + str(snr).replace('.', '_')
        except:
            pass
    return result


def random_select_data_as_noise(cur_item, all_files, safe_len=1, random_seed=None):
    # start_time = time.time()
    # random.seed(random_seed)
    # construct choice (data) lists - list of tuples
    # item[0]: video clip index
    # item[1]: data first bit's index in video clip
    # item[2]: data bit stream
    # item[3]: audio_path
    # item[4]: framerate
    cur_speaker = os.path.basename(get_parent_dir(cur_item[3]))
    # print(type(all_files))
    # print(len(all_files))
    chosen = random.choice(all_files)
    # i = np.random.randint(0, len(all_files))  # faster?
    # chosen = all_files[i]
    chosen_speaker = os.path.basename(get_parent_dir(chosen['path']))
    while cur_speaker == chosen_speaker or math.floor(chosen['duration']) < safe_len:
        chosen = random.choice(all_files)
        chosen_speaker = os.path.basename(get_parent_dir(chosen['path']))
    assert cur_speaker != chosen_speaker
    assert math.floor(chosen['duration']) >= safe_len
    # print(cur_speaker, chosen_speaker)
    # print("--- %s seconds in randselect ---" % (time.time() - start_time))
    # print('safe_len:', safe_len)
    return chosen['audio_path'], random.uniform(0, math.floor(chosen['duration']) - safe_len)  # -safe_len here because our data is less than safe_len seconds long


def random_select_data_as_noise_for_pred(cur_file, all_files, random_seed=None):
    results = []
    # start_time = time.time()
    # random.seed(random_seed)
    cur_speaker = os.path.basename(get_parent_dir(cur_file['path']))
    # print(type(all_files))
    # print(len(all_files))
    chosen = random.choice(all_files)
    # i = np.random.randint(0, len(all_files))  # faster?
    # chosen = all_files[i]
    chosen_speaker = os.path.basename(get_parent_dir(chosen['path']))
    while cur_speaker == chosen_speaker:
        chosen = random.choice(all_files)
        chosen_speaker = os.path.basename(get_parent_dir(chosen['path']))
    assert cur_speaker != chosen_speaker
    results.append(chosen)

    remaining_duration = cur_file['duration'] - chosen['duration']
    while remaining_duration > -1:  # just to be safe
        chosen = random.choice(all_files)
        chosen_speaker = os.path.basename(get_parent_dir(chosen['path']))
        while cur_speaker == chosen_speaker:
            chosen = random.choice(all_files)
            chosen_speaker = os.path.basename(get_parent_dir(chosen['path']))
        assert cur_speaker != chosen_speaker
        results.append(chosen)
        remaining_duration = remaining_duration - chosen['duration']
    # print(cur_speaker, chosen_speaker)
    # print("--- %s seconds in randselect ---" % (time.time() - start_time))
    # print('safe_len:', safe_len)
    return results


def random_select_noises_for_pred(cur_file, noises, random_seed=None):
    results = []
    # start_time = time.time()
    # random.seed(random_seed)
    # print(type(noises))
    # print(len(noises))
    chosen = random.choice(noises)
    # i = np.random.randint(0, len(noises))  # faster?
    # chosen = noises[i]
    results.append(chosen)

    remaining_duration = cur_file['audio_samples'] - len(chosen)
    while remaining_duration > -1:  # just to be safe
        chosen = random.choice(noises)
        results.append(chosen)
        remaining_duration = remaining_duration - len(chosen)
    # print("--- %s seconds in randselect ---" % (time.time() - start_time))

    return np.hstack(results)
