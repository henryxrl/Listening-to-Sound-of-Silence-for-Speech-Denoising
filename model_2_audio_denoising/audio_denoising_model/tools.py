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


JSON_DUMP_PARAMS = dict(indent=4, sort_keys=False, ensure_ascii=False, separators=(',', ':'))

PLOT_W = 50
PLOT_H = 10
PLOT_DPI = 100
PLOT_CHUNKSIZE = 1e10
Y_LIM_T = 1.2
Y_LIM_B = -1.2


def show_metrics(y_true, y_score, suppress_stdout=True):
    """Calculate different evaluation metrics"""
    y_true = np.int_(y_true)
    y_score = np.int_(y_score)

    # check number of 0/1 samples
    class_sample_count = dict([(int(t), len(np.where(y_true == t)[0]))\
        for t in np.unique(y_true)])

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
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2*tp / (2*tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    mcc = 0 if np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) == 0 else\
        (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

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

    # save stat results
    return OrderedDict([
        ('num_samples', len(y_true)),
        ('num_silent_samples', class_sample_count[0]),
        ('num_non_silent_samples', class_sample_count[1]),
        ('base', base),
        ('accuracy', accuracy),
        ('true_positive', int(tp)),
        ('false_positive', int(fp)),
        ('true_negative', int(tn)),
        ('false_negative', int(fn)),
        ('true_pos_rate(recall)', float(tpr)),
        ('false_pos_rate', float(fpr)),
        ('precision', float(precision)),
        ('true_neg_rate', float(tnr)),
        ('f1', float(f1)),
        ('roc_auc', float(auc)),
        ('mcc', float(mcc))
    ])


def truncate(l):
    """Truncate the leading and trailing '2' characters"""
    indices = [len(list(g)) for k, g in groupby(l) if k == '2']
    # return l[indices[0]:-indices[1]]
    return (indices[0], -indices[1])


def bit_stream_indices_list(files, data_len_sec, data_overlap_sec, random_seed=None, pred=False):
    """Create a list of truncated bit streams indices"""
    random.seed(random_seed)
    assert data_len_sec != data_overlap_sec

    lists = []
    for i, f in enumerate(files):
        # print(f['path'])

        try:
            idx1, idx2 = truncate(f['bit_stream'])
        except:
            idx1 = 0
            idx2 = len(f['bit_stream'])
        cur_clip_bs = f['bit_stream'][idx1:idx2]   # actual bits

        fps = f['framerate']
        start_sec = idx1 / fps
        end_sec = idx2 / fps

        if not pred:
            f_sr = float(f['audio_sample_rate'])
            f_len = float(f['audio_samples'])
            f_duration = min(float(f['duration']), f_len/f_sr, end_sec) - start_sec
            # print(f_duration)

            if f_duration < data_len_sec:
                continue

            f_num_data = math.floor((f_duration - data_len_sec) / (data_len_sec - data_overlap_sec)) + 1
            # print(f_num_data)

            f_start_pos = start_sec + np.arange(f_num_data) * (data_len_sec - data_overlap_sec) 
            # print(f_start_pos)

            for x in f_start_pos:
                # construct choice (data) lists - list of tuples
                # item[0]: video clip index
                # item[1]: data starting bit's index (in second) in video clip
                # item[2]: data ending bit's index (in second) in video clip
                # item[3]: data bit stream
                # item[4]: audio_path
                # item[5]: framerate
                choice = (i, x, x+data_len_sec, cur_clip_bs[int(x*fps):int((x+data_len_sec)*fps)], f['audio_path'], fps)
                lists.append(choice)
        else:
            choice = (i, start_sec, end_sec, cur_clip_bs, f['audio_path'], fps)
            lists.append(choice)

    return lists


def create_sample_list_from_indices(files, percent_samples_selected=None, data_len_sec=2, data_overlap_sec=1, random_seed=None, pred=False):
    """Create a list of samples from indices"""
    all_choices = bit_stream_indices_list(files, data_len_sec, data_overlap_sec, random_seed=random_seed, pred=pred)
    # each choice: tuple(file index, file's bitstream index, data bitstream, data center bit, bce weights)
    print('Total available samples: ', len(all_choices))

    if percent_samples_selected is None:
        return all_choices

    if percent_samples_selected > 1:
        percent_samples_selected = 1
    elif percent_samples_selected < 0:
        percent_samples_selected = 0

    np.random.seed(random_seed)
    all_chosen_indices = sorted(np.random.choice(len(all_choices), int(len(all_choices)*percent_samples_selected), replace=False))
    # print(all_chosen_indices)
    result = [all_choices[i] for i in all_chosen_indices]
    # print(result)
    return result


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
        # print('signal: ({}, {})'.format(np.amin(signal), np.amax(signal)))
        # print('signal len:', len(signal))
        # print('noise: ({}, {})'.format(np.amin(noise), np.amax(noise)))
        # print('noise len:', len(noise))
        # print('snr:', snr)
        # print('norm:', norm)
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
            # print('ratio:', ratio)
            # print('new_noise: ({}, {})'.format(np.amin(new_noise), np.amax(new_noise)))
            # print('new_noise len:', len(new_noise))
            # print('ret_signal: ({}, {})'.format(np.amin(ret_signal), np.amax(ret_signal)))
            # print('ret_signal len:', len(ret_signal))

    # print('IN ADD_SIGNALS')
    # print('clean_sig: ({}, {})'.format(np.amin(signal), np.amax(signal)))
    # print('mixed_sig: ({}, {})'.format(np.amin(ret_signal), np.amax(ret_signal)))
    # print('full_noise: ({}, {})'.format(np.amin(new_noises[0]), np.amax(new_noises[0])))
    # print('mixed == clean + full_noise?', np.array_equal(ret_signal, signal+new_noises[0]))
    # print('END')

    if norm:
        # print('normalize')
        scale = np.max(np.abs(ret_signal)) / norm
        if scale != 0:
            return ret_signal / scale, signal / scale, [x / scale for x in new_noises]
        else:
            return ret_signal, signal, new_noises
    else:
        # print('dont normalize')
        return ret_signal, signal, new_noises


def add_noise_to_audio(audio, noise, snr, start_pos=None, norm=None):
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

    # audio = librosa.util.normalize(audio)
    # noise_cropped = librosa.util.normalize(noise_cropped)
    # print('audio normalized: ({}, {})'.format(np.amin(audio), np.amax(audio)))
    # print('noise_cropped normalized: ({}, {})'.format(np.amin(noise_cropped), np.amax(noise_cropped)))
    # No need to normalize clean and noise signals because add_signals will calculate ratio 

    # create noisy signal input
    mixed_signal, clean_signal, noise_signals = add_signals(audio, [noise_cropped], snr=snr, norm=norm)

    return mixed_signal, clean_signal, noise_signals


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


def convert_threshold_to_suffix(threshold_str):
    result = ""
    if threshold_str != "":
        try:
            threshold = float(threshold_str)
            if 0 <= threshold <= 1:
                result = '_' + str(threshold).replace('.', '_')
        except:
            pass
    return result


def convert_bitstreammask_to_audiomask(ref_audio_signal, frames_to_audiosample_ratio, bitstream):
    mask = np.zeros_like(ref_audio_signal)
    for bit_idx, bit in enumerate(bitstream):
        # mask out non-silent intervals in ref_audio_signal
        # silent 1. non-silent 0
        if bit == '0':    # silent frame
            mask[int(bit_idx * frames_to_audiosample_ratio):int((bit_idx+1) * frames_to_audiosample_ratio - 1)] = 1
        elif bit == '1':  # non-silent frame
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


########## TEST: band pass filter ##########
from scipy.signal import butter, sosfiltfilt, sosfreqz


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = np.float32(sosfiltfilt(sos, data))
    return y
