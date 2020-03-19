import argparse
import json
import os
import random
import time
from itertools import groupby
from pathlib import Path

import cv2
import numpy as np
from pypesq import pesq
from scipy.interpolate import interp1d
from tqdm import tqdm

from common import (EXPERIMENT_DIR, PHASE_PREDICTION, PHASE_TESTING,
                    PHASE_TRAINING, PROJECT_ROOT, get_config)
from dataset import (DATA_LENGTH_SECONDS, DATA_OVERLAP_SECONDS,
                     DATA_REQUIRED_SR, DATA_ROOT, JSON_PARTIAL_NAME,
                     get_dataloader)
from metrics import evaluate_metrics
from networks import get_network
from tools import *
from transform import *
from utils import ensure_dir, get_parent_dir
from visualization import (draw_spectrum, draw_waveform,
                           draw_waveform_animated_better_quality,
                           draw_waveform_animated_faster)


BIT_STREAM_LABEL = 'recovered_prediction'
# BIT_STREAM_LABEL = 'bit_stream' # test ground true labeling on denoise effect
GT_BIT_STREAM_LABEL = 'bit_stream'

FIRST_MODEL_EXP_NAME = 'audioonly_model'
# FIRST_MODEL_EXP_NAME = 'audiovisual_model'
FIRST_MODEL_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "../model_1_silent_interval_detection/model_output", FIRST_MODEL_EXP_NAME, "outputs")

# UNKNOWN_CLEAN_SIGNAL_NAME = 'henrique_audioonly'
# UNKNOWN_CLEAN_SIGNAL_NAME = 'henrique_audiovisual'
# UNKNOWN_CLEAN_SIGNAL_NAME = 'languages_audioonly'
UNKNOWN_CLEAN_SIGNAL_NAME = 'looking_to_listen'
UNKNOWN_CLEAN_SIGNAL_FIRST_MODEL_OUTPUT_ROOT = os.path.join(FIRST_MODEL_OUTPUT_ROOT, UNKNOWN_CLEAN_SIGNAL_NAME)

EXPERIMENT_PREDICTION_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, 'outputs', FIRST_MODEL_EXP_NAME)


def interpolate_waveform(output, tlen):
    lineared = interp1d(np.arange(len(output)), output)
    steps = np.linspace(0, len(output) - 1, tlen)
    resampled_out = lineared(steps)
    return resampled_out


def metrics_L1(output, target):
    # L1 metrics
    lineared = interp1d(np.arange(len(output)), output)
    steps = np.linspace(0, len(output) - 1, len(target))
    resampled_out = lineared(steps)
    return np.mean(np.abs(resampled_out - target))


def metrics_pesq(output, target, sr=16000, mode='wb'):
    # pesq
    score = pesq(target, output, sr)
    return score


def metrics_SegSNR(output, target, sr=16000, frame_len=20):
    # Segmental SNR
    # FIXME : linear interpolation may cause misalignment
    lineared = interp1d(np.arange(len(output)), output)
    steps = np.linspace(0, len(output) - 1, len(target))
    output = lineared(steps)

    seg_size = int(sr * frame_len / 1000)
    n_segs = len(target) // seg_size
    remains = len(target) % seg_size

    # split the whole signal to segments
    target_segs = np.stack(np.split(target[:seg_size * n_segs], n_segs), axis=0)
    output_segs = np.stack(np.split(output[:seg_size * n_segs], n_segs), axis=0)

    result = 0
    # regular segments
    value = 10 * np.log10(np.sum(target_segs ** 2, axis=1) / (np.sum((target_segs - output_segs) ** 2, axis=1) + 1e-10) + 1e-10)
    result += np.sum(np.clip(value, -10, 35), axis=0)

    # remaining tail
    value = 10 * np.log(np.sum(target[-remains:] ** 2) / np.sum((target[-remains:] - output[-remains:]) ** 2 + 1e-10) + 1e-10)
    result += np.clip(value, -10, 35)

    result /= (n_segs + 1)
    return result


def get_data(random_seed, sr=16000, snr=2.5, n_fft=510, hop_length=158, win_length=400):
    data_info =        {
            "num_frames":268,
            "bit_stream":					"1111111111001101111100001111101100000000000000111111100011100111111111111111111111100011111000000000000000000111111111111111111111110111110000000000000000000100111111111111111100111100100111111100011111101111111111000000000000011011111111111111011111111110000111111111",
            "ground_truth_bit_stream":		"2222222111111111111111111111111111000000000011111111111111111111111111111111111111111111111110000000000000011111111111111111111111111111111100000000000000011111111111111111111111111111111111111111111111111111111111110000000001111111111111111111111111111111111112222222",
            "ground_truth_bit_stream_fixed":"2222222111111111111111111111111100000000000000111111111111111111111111111111111111111111111000000000000000000111111111111111111111111111110000000000000000000111111111111111111111111111111111111111111111111111111111000000000000011111111111111111111111111111111112222222",
            "predicted_bit_stream":			"2222222111111111111111111111111111111000000111111111111111111111111111111111111111111111111111111000000000111111111111111111111111111111110000000011000011111111111111111111111111111111111111111111111111111111111111111111111111111110001111111111111111111111111112222222",
            "predicted_bit_stream_fixed":	"2222222111111111111111111111111111100000000001111111111111111111111111111111111111111111111111100000000000001111111111111111111111111111000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111000000011111111111111111111111112222222"
        }
    audio_path = "audio_samples/clean/7qexZo5TZpE_180.079900.wav"
    NOISE_SRC_ROOT = "audio_samples/test_noise"
    ntypes = os.listdir(NOISE_SRC_ROOT)
    try:
        ntypes.remove('.DS_Store')
    except:
        pass
    print(ntypes)

    # load clean audio
    clean_signal, _ = librosa.load(audio_path, sr=sr)

    # randomly load noise audio and select an interval
    rng = random.Random(random_seed)
    noise_path = os.path.join(NOISE_SRC_ROOT, ntypes[rng.randint(0, len(ntypes) - 1)], "ch01.wav")
    full_noise_signal, _ = librosa.load(noise_path, sr=sr)
    start = rng.randint(0, len(full_noise_signal) - len(clean_signal))
    full_noise_signal = full_noise_signal[start:start + len(clean_signal)]

    # create noisy signal input
    mixed_signal, clean_signal, full_noise_signals = add_signals(clean_signal, [full_noise_signal], snr=snr)
    full_noise_signal = full_noise_signals[0]

    # convert video bitstream to audio mask
    mask = np.zeros_like(full_noise_signal)
    bitsteam = data_info['predicted_bit_stream_fixed']
    aframe2vframe = len(full_noise_signal) // data_info['num_frames']  # this will yield some accuracy loss
    for idx, ch in enumerate(bitsteam):
        if ch == '0':
            mask[idx * aframe2vframe: (idx + 1) * aframe2vframe] = 1
    noise_signal = mixed_signal * mask

    # stft
    mixed_sig_stft = fast_stft(mixed_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    clean_sig_stft = fast_stft(clean_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    noise_sig_stft = fast_stft(noise_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    full_noise_sig_stft = fast_stft(full_noise_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # icrm = fast_cRM_sigmoid(clean_sig_stft, mixed_sig_stft)

    mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    clean_sig_stft = torch.tensor(clean_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    noise_sig_stft = torch.tensor(noise_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    full_noise_sig_stft = torch.tensor(full_noise_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)

    return {
        "mixed": mixed_sig_stft, "clean": clean_sig_stft, "noise": noise_sig_stft, "full_noise": full_noise_sig_stft,
        "mask": mask, "clean_sig": clean_signal,
        "id": audio_path.split('/')[-1][:-4], 'snr': snr,
    }


def run(args, data_list, save_results=True, save_stat=False):
    # get config
    config = get_config(args)

    # get network
    net = get_network(config)
    name = args.ckpt if args.ckpt == 'latest' else "ckpt_epoch{}".format(args.ckpt)
    load_path = os.path.join("train_log/model", "{}.pth".format(name))
    net.load_state_dict(torch.load(load_path)['model_state_dict'])
    net = net.cuda()

    stat = []

    # forward
    net.eval()
    for data in tqdm(data_list):
        mixed_stft = data['mixed'].cuda()
        noise_stft = data['noise'].cuda()
        clean_stft = data['clean'].cuda()  # (B, 2, 257, L)
        full_noise_stft = data['full_noise'].numpy()[0].transpose((1, 2, 0))
        with torch.no_grad():
            pred_noise_stft, output_mask = net(mixed_stft, noise_stft)
            # output_mask = F.interpolate(output_mask, size=(257, output_mask.size(3)))

        mixed_stft = mixed_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        noise_stft = noise_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        clean_stft = clean_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        pred_noise_stft = pred_noise_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        output_mask = output_mask.detach().cpu().numpy()[0].transpose((1, 2, 0))

        output_stft = fast_icRM_sigmoid(mixed_stft, output_mask)

        mixed_sig = fast_istft(mixed_stft)
        noise_sig = fast_istft(noise_stft)
        clean_sig = fast_istft(clean_stft)
        output_sig = fast_istft(output_stft)
        pred_noise_sig = fast_istft(pred_noise_stft)
        full_noise_sig = fast_istft(full_noise_stft)

        cost_l1 = metrics_L1(output_sig, clean_sig)
        cost_pesq = metrics_pesq(output_sig, clean_sig)
        cost_ssnr = metrics_SegSNR(output_sig, clean_sig)
        info = {'id': str(data['id']),
                'l1': cost_l1,
                'pesq': cost_pesq,
                'ssnr': cost_ssnr,
                'snr': data['snr']}

        # save results
        if save_results:
            save_dir = os.path.join(args.outputs, str(data['id']))
            ensure_dir(save_dir)

            waveform = draw_waveform([mixed_sig, noise_sig, full_noise_sig, pred_noise_sig, clean_sig, output_sig])
            spectrum = draw_spectrum([mixed_sig, noise_sig, full_noise_sig, pred_noise_sig, clean_sig, output_sig])
            cv2.imwrite(os.path.join(save_dir, 'waveform.png'), waveform)
            cv2.imwrite(os.path.join(save_dir, 'spectrum.png'), spectrum)

            librosa.output.write_wav(os.path.join(save_dir, 'mixed-{}.wav'.format(data['snr'])), mixed_sig, config.sr)
            librosa.output.write_wav(os.path.join(save_dir, 'noise.wav'), noise_sig, config.sr)
            librosa.output.write_wav(os.path.join(save_dir, 'pred_noise.wav'), pred_noise_sig, config.sr)
            librosa.output.write_wav(os.path.join(save_dir, 'full_noise.wav'), full_noise_sig, config.sr)
            librosa.output.write_wav(os.path.join(save_dir, 'clean.wav'), clean_sig, config.sr)
            librosa.output.write_wav(os.path.join(save_dir, 'output.wav'), output_sig, config.sr)
            with open(os.path.join(save_dir, 'stat.json'), 'w') as fp:
                json.dump(info, fp)

        stat.append(info)

    if save_stat:
        avg_cost_l1 = sum([item['l1'] for item in stat]) / len(stat)
        avg_cost_pesq = sum([item['pesq'] for item in stat]) / len(stat)
        avg_cost_ssnr = sum([item['ssnr'] for item in stat]) / len(stat)
        stat = [{'avg_l1': avg_cost_l1, 'avg_pesq': avg_cost_pesq, 'avg_ssnr': avg_cost_ssnr}] + stat
        with open('eval_results.json', 'w') as fp:
            json.dump(stat, fp)


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
            mask[int(bit_idx * frames_to_audiosample_ratio):int((bit_idx+1) * frames_to_audiosample_ratio - 1)] = 0

    # check if mask has sporatic 0/1's
    mask_idx = 0
    for k, g in groupby(mask):
        g_len = len(list(g))
        if g_len < 5:
            mask[mask_idx:mask_idx+g_len] = 1 - k
        mask_idx += g_len
    # print(mask)
    return mask


def get_data_from_first_model(first_model_json_path, sr=16000, snr=2.5, n_fft=510, hop_length=158, win_length=400, unknown_clean_signal=False):
    # load prediction json
    all_data_info = []
    dataset_path = ''
    num_videos = 0
    data_total_frames = 0
    data_center_frames = 0
    sigmoid_threshold = 0
    with open(first_model_json_path, 'r') as fp:
        json_obj = json.load(fp)
        all_data_info = json_obj['files']
        dataset_path = json_obj['dataset_path']
        num_videos = json_obj['num_videos']
        data_total_frames = json_obj['data_total_frames']
        data_center_frames = json_obj['data_center_frames']
        sigmoid_threshold = json_obj['sigmoid_threshold']
        snr = json_obj['snr']

    print('========== SUMMARY ==========')
    print('Dataset JSON:', first_model_json_path)
    print('Dataset path:', dataset_path)
    print('Num videos:', num_videos)
    print('Sample rate: {}'.format(sr))
    print('SNR: {}'.format(snr))
    print('n_fft: {}'.format(n_fft))
    print('hop_length: {}'.format(hop_length))
    print('win_length: {}'.format(win_length))

    print('========== Loading data ==========')
    data_list = []
    for idx, data in enumerate(tqdm(all_data_info)):
        if not unknown_clean_signal:
            # load clean audio
            # clean_audio_path = data['path'].split('.mp4')[0] + '.wav'
            clean_audio_path = os.path.join(get_parent_dir(first_model_json_path), data['clean_audio'])
            clean_sig, _ = librosa.load(clean_audio_path, sr=sr)
            # clean_sig = librosa.util.normalize(clean_sig)

            # load full noise
            # full_noise_path = os.path.join(get_parent_dir(first_model_json_path),\
            #     'noise_' + os.path.splitext(os.path.basename(first_model_json_path))[0].split('_')[-1],\
            #         os.path.basename(clean_audio_path).split('.wav')[0] + '_noise.wav')
            full_noise_path = os.path.join(get_parent_dir(first_model_json_path), data['full_noise'])
            full_noise, _ = librosa.load(full_noise_path, sr=sr)
            # full_noise = librosa.util.normalize(full_noise)

        # load noisy audio
        mixed_audio_path = os.path.join(get_parent_dir(first_model_json_path), data['mixed_audio'])
        mixed_sig, _ = librosa.load(mixed_audio_path, sr=sr)

        # convert predicted bitstream to audio mask
        bitstream = data[BIT_STREAM_LABEL]
        gt_bitstream = data[GT_BIT_STREAM_LABEL]

        frames_to_audiosample_ratio = float(sr) / data['framerate']
        mask = convert_bitstreammask_to_audiomask(mixed_sig, frames_to_audiosample_ratio, bitstream)

        # create ground truth mask
        if not unknown_clean_signal:
            gt_mask = convert_bitstreammask_to_audiomask(clean_sig, frames_to_audiosample_ratio, gt_bitstream)

        # noise_sig(masked noise) = mixed_sig * mask
        noise_sig = mixed_sig * mask

        # stft
        mixed_sig_stft = fast_stft(mixed_sig, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        noise_sig_stft = fast_stft(noise_sig, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        if not unknown_clean_signal:
            # enforce silent intervals to be truly silent (clean_sig)
            clean_sig = clean_sig * (1 - gt_mask)
            clean_sig_stft = fast_stft(clean_sig, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            full_noise_sig_stft = fast_stft(full_noise, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # icrm = fast_cRM_sigmoid(clean_sig_stft, mixed_sig_stft)

        mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
        noise_sig_stft = torch.tensor(noise_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
        if not unknown_clean_signal:
            clean_sig_stft = torch.tensor(clean_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)
            full_noise_sig_stft = torch.tensor(full_noise_sig_stft.transpose((2, 0, 1)), dtype=torch.float32).unsqueeze(0)

        if unknown_clean_signal:
            item = OrderedDict([
                ("id", os.path.splitext(os.path.basename(data['path']))[0]),
                ("path", data['path']),
                ("mixed_audio_path", mixed_audio_path),
                ("bitstream", bitstream),
                ("mixed", mixed_sig_stft),
                ("noise", noise_sig_stft),
                ("mask", mask),
                ("snr", snr),
                ("sr", sr),
            ])
        else:
            item = OrderedDict([
                ("id", os.path.splitext(os.path.basename(data['path']))[0]),
                ("path", data['path']),
                ("clean_audio_path", clean_audio_path),
                ("mixed_audio_path", mixed_audio_path),
                ("full_noise_path", full_noise_path),
                ("bitstream", bitstream),
                ("mixed", mixed_sig_stft),
                ("clean", clean_sig_stft),
                ("noise", noise_sig_stft),
                ("full_noise", full_noise_sig_stft),
                ("mask", mask),
                ("snr", snr),
                ("sr", sr),
            ])
        data_list.append(item)

    info = OrderedDict([
        ('dataset_path', dataset_path),
        ('num_videos', num_videos),
        ('data_total_frames', data_total_frames),
        ('data_center_frames', data_center_frames),
        ('sigmoid_threshold', sigmoid_threshold)
    ])

    return (data_list, info)


def evaluate(args, data_list_info, save_individual_results=True, save_stat=True):
    print('========== Evaluating ==========')
    data_list = data_list_info[0]
    data_info = data_list_info[1]
    data_info['snr'] = args.snr
    # get config
    config = get_config(args)

    # get network
    net = get_network(config)
    name = args.ckpt if args.ckpt == 'latest' or args.ckpt == 'best_acc' else "ckpt_epoch{}".format(args.ckpt)
    load_path = os.path.join(config.model_dir, "{}.pth".format(name))
    print('Load saved model: {}'.format(load_path))
    net.load_state_dict(torch.load(load_path)['model_state_dict'])

    if torch.cuda.device_count() > 1:
        print('Multi-GPUs available')
        net = nn.DataParallel(net.cuda())   # For multi-GPU
    else:
        print('Single-GPU available')
        net = net.cuda()    # For single-GPU

    # evaluate
    net.eval()
    stat = []
    print('Save individual results:', save_individual_results)
    print('Save overall results:', save_stat)

    for i, data in enumerate(tqdm(data_list)):
        mixed_stft = data['mixed'].cuda()
        noise_stft = data['noise'].cuda()
        if not args.unknown_clean_signal:
            clean_stft = data['clean'].cuda()  # (B, 2, 257, L)
            full_noise_stft = data['full_noise'].numpy()[0].transpose((1, 2, 0))
        with torch.no_grad():
            pred_noise_stft, output_mask = net(mixed_stft, noise_stft)
            # output_mask = F.interpolate(output_mask, size=(257, output_mask.size(3)))

        mixed_stft = mixed_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        noise_stft = noise_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        if not args.unknown_clean_signal:
            clean_stft = clean_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        pred_noise_stft = pred_noise_stft.detach().cpu().numpy()[0].transpose((1, 2, 0))
        output_mask = output_mask.detach().cpu().numpy()[0].transpose((1, 2, 0))

        output_stft = fast_icRM_sigmoid(mixed_stft, output_mask)

        mixed_sig = fast_istft(mixed_stft)
        noise_sig = fast_istft(noise_stft)
        output_sig = fast_istft(output_stft)

        # check if mask has sporatic 0/1's
        # groups = []
        # for k, g in groupby(data['mask']):
        #     groups.append((str(int(k)), len(list(g))))      # Store group iterator as a list
        # print(groups)

        # TEST: suppress silent intervals to be truly silent
        # output_sig = interpolate_waveform(output_sig, len(data['mask']))  # make output_sig same length as mask
        # output_sig = np.multiply(output_sig, 1 - data['mask'])  # make all silent intervals silent
        # TEST done

        ##### TEST: for non-silent intervals, get their low and high frequency bands #####
        # output_sig = butter_bandpass_filter(output_sig, 300, 3400, data['sr'])
        ##### TEST done #####

        pred_noise_sig = fast_istft(pred_noise_stft)
        if not args.unknown_clean_signal:
            clean_sig = fast_istft(clean_stft)
            # output_sig = interpolate_waveform(output_sig, len(clean_sig))  # from TEST: make output_sig same length as clean_sig
            full_noise_sig = fast_istft(full_noise_stft)[:len(mixed_sig)]

        if not args.unknown_clean_signal:
            info = OrderedDict([
                ('id', str(data['id'])),
                ('path', str(data['path'])),
                ('clean_audio_path', data['clean_audio_path']),
                ('mixed_audio_path', data['mixed_audio_path']),
                ('full_noise_path', data['full_noise_path']),
                ('bitstream', data['bitstream']),
                ('sr', data['sr']),
                ('snr', data['snr']),
            ])

            # resample audios to 16000 for metrics calculations
            output_sig_16k = librosa.resample(output_sig, data['sr'], 16000)
            clean_sig_16k = librosa.resample(clean_sig, data['sr'], 16000)
            
            # calculate metrics
            info.update(evaluate_metrics(output_sig_16k, clean_sig_16k, sr=16000))
        else:
            info = OrderedDict([
                ('id', str(data['id'])),
                ('path', str(data['path'])),
                ('mixed_audio_path', data['mixed_audio_path']),
                ('bitstream', data['bitstream']),
                ('sr', data['sr']),
                ('snr', data['snr']),
            ])

        # save results
        if save_individual_results:
            save_dir = os.path.join(os.path.abspath(args.outputs), convert_snr_to_suffix2(args.snr)[1:], str(data['id']))
            ensure_dir(save_dir)

            if not args.unknown_clean_signal:
                waveform = draw_waveform([mixed_sig, noise_sig, full_noise_sig, pred_noise_sig, clean_sig, output_sig], sr=data['sr'],\
                    titles=['Noisy Input', 'Noise Intervals', 'Ground Truth Full Noise', 'Predicted Full Noise', 'Ground Truth Clean Input', 'Denoised Output'])
                spectrum = draw_spectrum([mixed_sig, noise_sig, full_noise_sig, pred_noise_sig, clean_sig, output_sig], sr=data['sr'],\
                    titles=['Noisy Input', 'Noise Intervals', 'Ground Truth Full Noise', 'Predicted Full Noise', 'Ground Truth Clean Input', 'Denoised Output'])
                waveform_path = os.path.join(save_dir, 'waveform.png')
                cv2.imwrite(waveform_path, waveform)
                spectrum_path = os.path.join(save_dir, 'spectrum.png')
                cv2.imwrite(spectrum_path, spectrum)
                info['waveform'] = waveform_path
                info['spectrum'] = spectrum_path

                # draw_waveform_animated_faster(os.path.join(save_dir, 'waveform.mp4'),\
                #     [mixed_sig, noise_sig, full_noise_sig, pred_noise_sig, clean_sig, output_sig], sr=data['sr'],\
                #         titles=['Noisy Input', 'Noise Intervals', 'Ground Truth Full Noise', 'Predicted Full Noise', 'Ground Truth Clean Input', 'Denoised Output'],\
                #             fps=30)
            else:
                waveform = draw_waveform([mixed_sig, noise_sig, pred_noise_sig, output_sig], sr=data['sr'],\
                    titles=['Noisy Input', 'Noise Intervals', 'Predicted Full Noise', 'Denoised Output'])
                spectrum = draw_spectrum([mixed_sig, noise_sig, pred_noise_sig, output_sig], sr=data['sr'],\
                    titles=['Noisy Input', 'Noise Intervals', 'Predicted Full Noise', 'Denoised Output'])
                waveform_path = os.path.join(save_dir, 'waveform.png')
                cv2.imwrite(waveform_path, waveform)
                spectrum_path = os.path.join(save_dir, 'spectrum.png')
                cv2.imwrite(spectrum_path, spectrum)
                info['waveform'] = waveform_path
                info['spectrum'] = spectrum_path

                # draw_waveform_animated_faster(os.path.join(save_dir, 'waveform.mp4'),\
                #     [mixed_sig, noise_sig, pred_noise_sig, output_sig], sr=data['sr'],\
                #         titles=['Noisy Input', 'Noise Intervals', 'Predicted Full Noise', 'Denoised Output'],\
                #             fps=30)

            noisy_input_path = os.path.join(save_dir, 'noisy_input.wav')
            librosa.output.write_wav(noisy_input_path, mixed_sig, data['sr'])
            noise_intervals_path = os.path.join(save_dir, 'noise_intervals.wav')
            librosa.output.write_wav(noise_intervals_path, noise_sig, data['sr'])
            predicted_full_noise_path = os.path.join(save_dir, 'predicted_full_noise.wav')
            librosa.output.write_wav(predicted_full_noise_path, pred_noise_sig, data['sr'])
            denoised_output_path = os.path.join(save_dir, 'denoised_output.wav')
            librosa.output.write_wav(denoised_output_path, output_sig, data['sr'])
            info['noisy_input'] = noisy_input_path
            info['noise_intervals'] = noise_intervals_path
            info['predicted_full_noise'] = predicted_full_noise_path
            info['denoised_output'] = denoised_output_path

            if not args.unknown_clean_signal:
                gt_full_noise_path = os.path.join(save_dir, 'ground_truth_full_noise.wav')
                librosa.output.write_wav(gt_full_noise_path, full_noise_sig, data['sr'])
                gt_clean_input_path = os.path.join(save_dir, 'ground_truth_clean_input.wav')
                librosa.output.write_wav(gt_clean_input_path, clean_sig, data['sr'])
                info['ground_truth_full_noise'] = gt_full_noise_path
                info['ground_truth_clean_input'] = gt_clean_input_path
            
            with open(os.path.join(save_dir, 'stat.json'), 'w') as fp:
                json.dump(info, fp, **JSON_DUMP_PARAMS)

        stat.append(info)

    if save_stat:
        if not args.unknown_clean_signal:
            avg_cost_l1 = sum([item['l1'] for item in stat]) / len(stat)
            avg_cost_stoi = sum([item['stoi'] for item in stat]) / len(stat)
            avg_cost_csig = sum([item['csig'] for item in stat]) / len(stat)
            avg_cost_cbak = sum([item['cbak'] for item in stat]) / len(stat)
            avg_cost_covl = sum([item['covl'] for item in stat]) / len(stat)
            avg_cost_pesq = sum([item['pesq'] for item in stat]) / len(stat)
            avg_cost_ssnr = sum([item['ssnr'] for item in stat]) / len(stat)

            data_info['denoise_statistics'] = OrderedDict([
                ('avg_l1', avg_cost_l1),
                ('avg_stoi', avg_cost_stoi),
                ('avg_csig', avg_cost_csig),
                ('avg_cbak', avg_cost_cbak),
                ('avg_covl', avg_cost_covl),
                ('avg_pesq', avg_cost_pesq),
                ('avg_ssnr', avg_cost_ssnr)
            ])

        data_info['files'] = stat

        suffix = convert_threshold_to_suffix(args.threshold)
        nsuffix = convert_snr_to_suffix2(args.snr)
        save_stat_path = os.path.join(os.path.abspath(args.outputs), 'eval_results' + suffix + nsuffix + '.json')
        with open(save_stat_path, 'w') as fp:
            json.dump(data_info, fp, **JSON_DUMP_PARAMS)


def main():
    print('Getting results from \'{}\''.format(FIRST_MODEL_EXP_NAME))

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('-o', '--outputs', default=EXPERIMENT_PREDICTION_OUTPUT_DIR, type=str, help="outputs dir to write results")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--eval', action='store_true', default=True, help="evaluate whole dataset")
    parser.add_argument('--gen', action='store_true', default=False, help="generate whole dataset pred noise")
    parser.add_argument('-t', '--threshold', type=str, default="", required=False, help="specify threshold used")
    parser.add_argument('--snr', type=float, default=None, required=False)
    parser.add_argument('--save_results', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)
    parser.add_argument('--unknown_clean_signal', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)
    args = parser.parse_args()

    if not args.unknown_clean_signal and args.snr is None:
        parser.error("--unknown_clean_signal [False] REQUIRES --snr")

    if not args.eval:
        data_list = [get_data(time.time())]
        run(args, data_list)
    else:
        suffix = convert_threshold_to_suffix(args.threshold)
        input_json = 'pred_data.json'.split('.json')[0] + '{}.json'.format(suffix)
        nsuffix = convert_snr_to_suffix2(args.snr)
        input_json = input_json.split('.json')[0] + '{}.json'.format(nsuffix)
        if not args.unknown_clean_signal:
            first_model_json_path = os.path.join(FIRST_MODEL_OUTPUT_ROOT, input_json)
            # print(first_model_json_path)
            # print(os.path.exists(first_model_json_path))
            data_list_info = get_data_from_first_model(first_model_json_path, sr=DATA_REQUIRED_SR, snr=args.snr)
            evaluate(args, data_list_info, save_individual_results=args.save_results, save_stat=True)
        else:
            args.outputs = os.path.join(EXPERIMENT_DIR, 'outputs', UNKNOWN_CLEAN_SIGNAL_NAME + '_' + FIRST_MODEL_EXP_NAME)
            ensure_dir(args.outputs)
            first_model_json_path = os.path.join(UNKNOWN_CLEAN_SIGNAL_FIRST_MODEL_OUTPUT_ROOT, input_json)
            # print(first_model_json_path)
            # print(os.path.exists(first_model_json_path))
            # print(args.outputs)
            data_list_info = get_data_from_first_model(first_model_json_path, sr=DATA_REQUIRED_SR, snr=args.snr, unknown_clean_signal=args.unknown_clean_signal)
            evaluate(args, data_list_info, save_individual_results=args.save_results, save_stat=True)


if __name__ == '__main__':
    main()
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python3 predict.py --unknown_clean_signal true
