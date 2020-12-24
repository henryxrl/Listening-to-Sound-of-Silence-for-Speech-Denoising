import argparse
import copy
import functools
import json
import os
from collections import OrderedDict

import librosa
import numpy as np
import torch
from tqdm import tqdm

from common import (EXPERIMENT_DIR, PHASE_PREDICTION, PHASE_TESTING,
                    PHASE_TRAINING, get_config)
from dataset import (CLIP_FRAMES, DATA_MAX_AUDIO_SAMPLES, DATA_REQUIRED_FPS,
                     DATA_REQUIRED_SR, DATA_ROOT, JSON_PARTIAL_NAME,
                     RANDOM_SEED, SILENT_CONSECUTIVE_FRAMES, SNRS,
                     get_dataloader)
from networks import get_network
from tools import *
from utils import ensure_dir, get_parent_dir


# Use multiple GPUs
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python3 predict.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 python3 predict.py
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 python3 predict.py


SIGMOID_THRESHOLD = 0.5
EXPERIMENT_PREDICTION_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, 'outputs')

SOUND_OF_SILENCE_JSON = os.path.join(DATA_ROOT, 'sounds_of_silence.json')
TIMIT_JSON = '/proj/vondrick/rx2132/test_noise_robust_embedding/data/TIMIT/TEST_noisy_snr10.json'
EXTERNAL_DATASET_JSON = TIMIT_JSON


def evaluate(args, save_individual_results=True, save_noise_info=True, save_stat=True, dataset_json=None, clean_audio=True):
    # all_ids = list(range(len(ALL_CHOICES)))
    # run(args, all_ids, save_results=save_results, save_stat=save_stat, is_ids=True)

    # if dataset_json is not None:
    if not clean_audio:
        args.outputs = os.path.join(EXPERIMENT_PREDICTION_OUTPUT_DIR, os.path.basename(dataset_json).split('.json')[0])
        ensure_dir(args.outputs)
        print('Output redirected to: \"{}\"'.format(args.outputs))

    # get config
    config = get_config(args)

    # get network
    net = get_network()
    name = args.ckpt if args.ckpt == 'latest' or args.ckpt == 'best_acc' else "ckpt_epoch{}".format(args.ckpt)
    load_path = os.path.join(config.model_dir, "{}.pth".format(name))
    print('Load saved model: {}'.format(load_path))
    net.load_state_dict(torch.load(load_path)['model_state_dict'])

    if torch.cuda.device_count() > 1:
        print('For multi-GPU')
        net = nn.DataParallel(net.cuda())   # For multi-GPU
    else:
        print('For single-GPU')
        net = net.cuda()    # For single-GPU

    # evaluate
    net.eval()
    stat = []
    print('Save individual results:', save_individual_results)
    print('Save overall results:', save_stat)
    # dataloader = get_dataloader(PHASE_PREDICTION, batch_size=config.batch_size, num_workers=config.num_workers, snr_idx=args.snr_idx)
    dataloader = get_dataloader(PHASE_PREDICTION, batch_size=1, num_workers=20, snr_idx=args.snr_idx, dataset_json=dataset_json, clean_audio=clean_audio)
    if clean_audio:
        suffix = convert_snr_to_suffix(dataloader.dataset.snrs, args.snr_idx)
    else:
        # suffix = convert_snr_to_suffix(SNRS, args.snr_idx)
        suffix = ''

    # save noise info
    if clean_audio:
        if save_noise_info:
            noise_dir = os.path.join(os.path.abspath(args.outputs), 'noise' + suffix)
            ensure_dir(noise_dir)
            noise_infos = OrderedDict([
                ('snrs', dataloader.dataset.snrs)
            ])
            noise_files = OrderedDict()
            noise_dict = dataloader.dataset.noise_dict
            for k, v in noise_dict.items():
                corr_video = os.path.basename(dataloader.dataset.files[k]['path'])
                noise_name = corr_video.split('.mp4')[0] + '_noise.wav'
                noise_path = os.path.join(noise_dir, noise_name)
                librosa.output.write_wav(noise_path, v[0], DATA_REQUIRED_SR)

                noise_files[corr_video] = OrderedDict([
                    ('audio', corr_video.split('.mp4')[0] + '.wav'),
                    ('noise', noise_name),
                    ('snr', v[1])
                ])
            noise_infos['files'] = noise_files
            # noise_json_path = os.path.join(noise_dir, 'noise.json')
            noise_json_path = os.path.join(noise_dir, suffix[1:]+'.json')
            with open(noise_json_path, 'w') as fpnoise:
                json.dump(noise_infos, fpnoise, **JSON_DUMP_PARAMS)
            print('Noise info saved to: \'{}\''.format(noise_json_path))

    for i, data in enumerate(tqdm(dataloader)):
        # batch_frames = data['frames'].cuda()
        # print('batch_frames.shape:', batch_frames.shape)
        # batch_frames_raws = data['frames'].detach().cpu()
        batch_labels = data['label'].detach().cpu().numpy()
        # print('batch_labels.shape:', batch_labels.shape)
        batch_audios = data['audio'].cuda()
        batch_audio_raws = data['audio'].detach().cpu().numpy()
        # print('batch_audio_raws.shape:', batch_audio_raws.shape)

        with torch.no_grad():
            batch_output_values = torch.sigmoid(net(s=batch_audios, v_num_frames=batch_labels.shape[1])).detach().cpu().numpy()
            # print('batch_output_values.shape:', batch_output_values.shape)
            batch_pred_labels = (batch_output_values >= SIGMOID_THRESHOLD).astype(np.float)
            # print('batch_pred_labels.shape:', batch_pred_labels.shape)

            for idx in range(batch_output_values.shape[0]):
                data_id = i * dataloader.batch_size + idx

                item = dataloader.dataset.items[data_id]
                file_info_dict = dataloader.dataset.files[item[0]]

                # print(batch_labels[idx].shape)
                label = list(batch_labels[idx].astype(np.int).astype(str))
                # print(len(label))
                pred_label = list(batch_pred_labels[idx].astype(np.int).astype(str))
                # print(len(pred_label))

                info = OrderedDict([
                    ('id', data_id),
                    ('path', file_info_dict['path']),
                    ('full_bit_stream', file_info_dict[BITSTREAM_JSON_LABEL]),
                    ('num_frames', file_info_dict['num_frames']),
                    ('framerate', file_info_dict['framerate']),
                    ('audio_sample_rate', file_info_dict['audio_sample_rate']),
                    ('audio_samples', file_info_dict['audio_samples']),
                    ('duration', file_info_dict['duration']),
                    ('frame_start_idx', item[1]),
                    ('label', label),
                    ('pred_label', pred_label),
                    ('match', (label == pred_label)),
                    ('confidence', list(batch_output_values[idx].astype(str)))
                ])
                stat.append(info)

                # save results
                if save_individual_results:
                    # only save interesting results
                    if label != pred_label or label == 0.0 or pred_label == 0.0:
                        # save_dir = os.path.join(os.path.abspath(args.outputs), 'frames' + suffix, str(data_id))
                        # ensure_dir(save_dir)

                        # with open(os.path.join(save_dir, 'stat.json'), 'w') as fp:
                        #     json.dump(info, fp, **JSON_DUMP_PARAMS)

                        # save images to visualize input
                        # frame_images = save_imgs_from_tensor(batch_frames_raws[idx], save_dir)

                        # save gif
                        gif_path = os.path.join(os.path.abspath(args.outputs), 'gifs' + suffix, str(data_id)+'.jpg')
                        ensure_dir(get_parent_dir(gif_path))
                        info['gif'] = os.path.join('gifs' + suffix, str(data_id)+'.jpg') # need relative path!
                        # create_gif3(frame_images, gif_path, fps=DATA_REQUIRED_FPS)
                        plt.figure(figsize=(PLOT_H, PLOT_H), dpi=PLOT_DPI)
                        # librosa.display.waveplot(np.squeeze(batch_audio_raws[idx]), sr=DATA_REQUIRED_SR)
                        mixed_stft = batch_audio_raws[idx].transpose((1, 2, 0))
                        mixed_sig = fast_istft(mixed_stft)
                        librosa.display.waveplot(mixed_sig, sr=DATA_REQUIRED_SR)
                        plt.savefig(gif_path)

                        # save audio
                        audio_path = os.path.join(os.path.abspath(args.outputs), 'audio' + suffix, str(data_id)+'.wav')
                        ensure_dir(get_parent_dir(audio_path))
                        info['audio'] = os.path.join('audio' + suffix, str(data_id)+'.wav')
                        # librosa.output.write_wav(audio_path, np.squeeze(batch_audio_raws[idx]), DATA_REQUIRED_SR)
                        # mixed_stft = batch_audio_raws[idx].transpose((1, 2, 0))
                        # mixed_sig = fast_istft(mixed_stft)
                        librosa.output.write_wav(audio_path, mixed_sig, DATA_REQUIRED_SR)

    if save_stat:
        # labels = [item['label'] for item in stat]
        # pred_labels = [item['pred_label'] for item in stat]
        # stat_dict = show_metrics(labels, pred_labels)

        stat_dict = OrderedDict()

        stat_dict['data_total_frames'] = CLIP_FRAMES
        stat_dict['data_center_frames'] = SILENT_CONSECUTIVE_FRAMES
        stat_dict['sigmoid_threshold'] = SIGMOID_THRESHOLD
        if args.snr_idx is not None:
            stat_dict['snr'] = dataloader.dataset.snrs[args.snr_idx]
            # stat_dict['snr'] = SNRS[args.snr_idx]
        else:
            stat_dict['snr'] = None

        # labels_all = [item['label'] for item in stat]
        labels_all = [element for item in stat for element in item['label']]
        # pred_labels_all = [item['pred_label'] for item in stat]
        pred_labels_all = [element for item in stat for element in item['pred_label']]
        # labels_partial = [item['label'] for item in stat\
        #     if item['bit_stream'].count('0') == len(item['bit_stream'])\
        #             or item['bit_stream'].count('1') > len(item['bit_stream']) - MAX_NUM_OF_0_IN_1_DATA]
        # pred_labels_partial = [item['pred_label'] for item in stat\
        #     if item['bit_stream'].count('0') == len(item['bit_stream'])\
        #             or item['bit_stream'].count('1') > len(item['bit_stream']) - MAX_NUM_OF_0_IN_1_DATA]
        # stat_dict['prediction_statistics'] = OrderedDict([
        #     (str(TRAIN_DATA_RATIO) + '-' + str(1-TRAIN_DATA_RATIO), show_metrics(labels_partial, pred_labels_partial)),
        #     (str(PRED_DATA_RATIO) + '-' + str(1-PRED_DATA_RATIO), show_metrics(labels_all, pred_labels_all))
        # ])
        # stat_dict['prediction_statistics'] = OrderedDict([
        #     ('filtered', show_metrics(labels_all, pred_labels_all)),
        #     # (str(PRED_DATA_RATIO) + '-' + str(1-PRED_DATA_RATIO), show_metrics(labels_all, pred_labels_all))
        # ])
        stat_dict['prediction_statistics'] = OrderedDict([
            # ('filtered', show_metrics(labels_partial, pred_labels_partial)),
            ('all', show_metrics(labels_all, pred_labels_all))
        ])

        # sort stat by confidence descending order
        stat = sorted(stat, key=lambda x:np.mean([float(i) for i in x['confidence']]), reverse=True)
        stat_dict['data'] = stat

        # save_stat_path = os.path.abspath('eval_results.json')
        save_stat_path = os.path.join(os.path.abspath(args.outputs), 'eval_results' + suffix + '.json')
        ensure_dir(get_parent_dir(save_stat_path))
        with open(save_stat_path, 'w') as fp:
            json.dump(stat_dict, fp, **JSON_DUMP_PARAMS)
        print('Overall results saved to: \'{}\''.format(save_stat_path))


def get_data_by_id(id=None):
    if id is None:
        id = np.random.randint(0, len(ALL_CHOICES))
    else:
        id = len(ALL_CHOICES)-1 if id >= len(ALL_CHOICES) else id

    item = ALL_CHOICES[id]
    file_info_dict = FILES[item[0]]
    assert item[1]+CLIP_FRAMES <= int(file_info_dict['num_frames'])

    try:
        # Get labels
        # labels = file_info_dict['bit_stream'][item[1]:item[1]+CLIP_FRAMES]
        labels = item[2]

        # Get center label
        # label = torch.tensor(float(get_centerbit(labels, SILENT_CONSECUTIVE_FRAMES)),\
        # dtype=torch.float32).unsqueeze(-1)    # Give the center label
        # label = float(get_centerbit(labels, SILENT_CONSECUTIVE_FRAMES))    # Give the center label
        label = float(item[3])

        # Get weights
        # weights = torch.tensor(item[4], dtype=torch.float32).unsqueeze(0)

        # Get frames
        frames = torch.stack(list(\
            map(functools.partial(load_image_from_index, dir_path=file_info_dict['frames_path']),\
                range(item[1], item[1]+CLIP_FRAMES)))).permute(1, 0, 2, 3)
        frames = center_crop_batch(frames).unsqueeze(0)
        # frames = frames.unsqueeze(0)

        # Get audio
        snd, sr = librosa.load(item[5], sr=DATA_REQUIRED_SR)
        # snd = librosa.util.normalize(snd)
        # snd = audio_normalize(snd)

        # get corresponding audio chunk
        audio_raw = snd[int(item[1]/item[6]*sr):int((item[1]+CLIP_FRAMES)/item[6]*sr)]
        audio = audio_raw[:DATA_MAX_AUDIO_SAMPLES]
        diff = DATA_MAX_AUDIO_SAMPLES - len(audio)
        if diff > 0:
            audio = np.concatenate((audio, np.zeros(diff)))
        audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    except Exception as e:
        # print(e)
        raise RuntimeError

    return {
        "path": file_info_dict['path'],
        "full_bit_stream": file_info_dict[BITSTREAM_JSON_LABEL],
        "num_frames": file_info_dict['num_frames'],
        "frame_start_idx": item[1],
        "frames" : frames,
        "bits": labels,
        "label": label,
        "audio": audio,
        "audio_raw": audio_raw
    }


def run(args, data_list, save_results=True, save_stat=False, is_ids=False):
    # get config
    config = get_config(args)

    # get network
    net = get_network()
    name = args.ckpt if args.ckpt == 'latest' or args.ckpt == 'best_acc' else "ckpt_epoch{}".format(args.ckpt)
    load_path = os.path.join(config.model_dir, "{}.pth".format(name))
    print('Load saved model: {}'.format(load_path))
    net.load_state_dict(torch.load(load_path)['model_state_dict'])
    net = net.cuda()

    stat = []

    # forward
    net.eval()
    print('Save individual result:', save_results)
    print('Save overall results:', save_stat)
    print('Num test samples:', len(data_list))
    for data_id, data in enumerate(tqdm(data_list)):
        if is_ids:
            data = get_data_by_id(data)
        frames = data['frames']
        label = data['label']
        audio = data['audio'].cuda()
        audio_raw = data['audio_raw']

        with torch.no_grad():
            output_value = net(audio).detach().cpu().numpy()
            output_value = float(np.squeeze(sigmoid(output_value)))
            pred_label = 1.0 if output_value >= SIGMOID_THRESHOLD else 0.0
            # pred_label = (torch.sigmoid(net(audio)).detach().cpu().numpy() >= SIGMOID_THRESHOLD).astype(np.float)

        info = OrderedDict([
            ('id', data_id),
            ('path', data['path']),
            ('full_bit_stream', data['full_bit_stream']),
            ('num_frames', data['num_frames']),
            ('framerate', data['framerate']),
            ('frame_start_idx', data['frame_start_idx']),
            ('bit_stream', data['bits']),
            ('label', label),
            ('pred_label', pred_label),
            ('match', (label == pred_label)),
            ('confidence', output_value)
        ])

        # save results
        if save_results:
            # only save interesting results
            if label != pred_label or label == 0.0 or pred_label == 0.0:
                save_dir = os.path.join(os.path.abspath(args.outputs), 'frames', str(data_id))
                ensure_dir(save_dir)

                with open(os.path.join(save_dir, 'stat.json'), 'w') as fp:
                    json.dump(info, fp, **JSON_DUMP_PARAMS)

                # save images to visualize input
                frame_images = save_imgs_from_tensor(frames, save_dir)

                # save gif
                gif_path = os.path.join(os.path.abspath(args.outputs), 'gifs', str(data_id)+'.gif')
                ensure_dir(get_parent_dir(gif_path))
                info['gif'] = os.path.join('gifs', str(data_id)+'.gif') # need relative path!
                create_gif3(frame_images, gif_path, fps=DATA_REQUIRED_FPS)

                # save audio
                audio_path = os.path.join(os.path.abspath(args.outputs), 'audio', str(data_id)+'.wav')
                ensure_dir(get_parent_dir(audio_path))
                info['audio'] = os.path.join('audio', str(data_id)+'.wav')
                librosa.output.write_wav(audio_path, audio_raw, DATA_REQUIRED_SR)

        stat.append(info)

    if save_stat:
        # labels = [item['label'] for item in stat]
        # pred_labels = [item['pred_label'] for item in stat]
        # stat_dict = show_metrics(labels, pred_labels)

        stat_dict = OrderedDict()

        stat_dict['data_total_frames'] = CLIP_FRAMES
        stat_dict['data_center_frames'] = SILENT_CONSECUTIVE_FRAMES
        stat_dict['sigmoid_threshold'] = SIGMOID_THRESHOLD

        labels_all = [item['label'] for item in stat]
        pred_labels_all = [item['pred_label'] for item in stat]
        labels_partial = [item['label'] for item in stat\
            if item['bit_stream'].count('0') == len(item['bit_stream'])\
                    or item['bit_stream'].count('1') > len(item['bit_stream']) - MAX_NUM_OF_0_IN_1_DATA]
        pred_labels_partial = [item['pred_label'] for item in stat\
            if item['bit_stream'].count('0') == len(item['bit_stream'])\
                    or item['bit_stream'].count('1') > len(item['bit_stream']) - MAX_NUM_OF_0_IN_1_DATA]
        # stat_dict['prediction_statistics'] = OrderedDict([
        #     (str(TRAIN_DATA_RATIO) + '-' + str(1-TRAIN_DATA_RATIO), show_metrics(labels_partial, pred_labels_partial)),
        #     (str(PRED_DATA_RATIO) + '-' + str(1-PRED_DATA_RATIO), show_metrics(labels_all, pred_labels_all))
        # ])
        # stat_dict['prediction_statistics'] = OrderedDict([
        #     ('filtered', show_metrics(labels_all, pred_labels_all)),
        #     # (str(PRED_DATA_RATIO) + '-' + str(1-PRED_DATA_RATIO), show_metrics(labels_all, pred_labels_all))
        # ])
        stat_dict['prediction_statistics'] = OrderedDict([
            ('filtered', show_metrics(labels_partial, pred_labels_partial)),
            ('all', show_metrics(labels_all, pred_labels_all))
        ])

        # sort stat by confidence descending order
        stat = sorted(stat, key=lambda x:x['confidence'], reverse=True)
        stat_dict['data'] = stat

        # save_stat_path = os.path.abspath('eval_results.json')
        save_stat_path = os.path.join(os.path.abspath(args.outputs), 'eval_results.json')
        ensure_dir(get_parent_dir(save_stat_path))
        with open(save_stat_path, 'w') as fp:
            json.dump(stat_dict, fp, **JSON_DUMP_PARAMS)
        print('Overall results saved to: \'{}\''.format(save_stat_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('-o', '--outputs', default=EXPERIMENT_PREDICTION_OUTPUT_DIR, type=str, help="outputs dir to write results")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    parser.add_argument('--eval', action='store_true', default=True, help="evaluate whole dataset")
    parser.add_argument('--gen', action='store_true', default=False, help="generate whole dataset pred noise")
    parser.add_argument('--snr_idx', type=int, default=None)
    parser.add_argument('--save_results', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)
    parser.add_argument('--unknown_clean_signal', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)
    args = parser.parse_args()
    # print(args)

    if not args.eval:
        # OBSOLETE - This was used to test single data;
        # Please use the "eval" mode instead, which is default to be turned ON
        data_json = os.path.join(DATA_ROOT, PHASE_PREDICTION + JSON_PARTIAL_NAME)
        print('Prediction data json:', data_json)
        with open(data_json, 'r') as fp:
            info = json.load(fp)
        global FILES
        FILES = info['files']

        global ALL_CHOICES
        ALL_CHOICES = create_sample_list_from_indices(info['files'], clip_frames=CLIP_FRAMES, silent_consecutive_frames=SILENT_CONSECUTIVE_FRAMES, random_seed=RANDOM_SEED, pred=True)

        # check number of 0/1 samples
        all_data_labels = [x[3] for x in ALL_CHOICES]
        class_sample_count = dict([('Class \'{}\''.format(str(int(t))),\
            len(np.where(all_data_labels == t)[0]))\
                for t in np.unique(all_data_labels)])
        print("Class_sample_count:", class_sample_count)

        if args.id is not None:
            data_list = [get_data_by_id(int(args.id))]
        else:
            data_list = [get_data_by_id() for i in range(args.num)]
        run(args, data_list)
    else:
        if not args.unknown_clean_signal:
            evaluate(args, save_individual_results=args.save_results, save_noise_info=True, save_stat=True, clean_audio=True)
        else:
            evaluate(args, save_individual_results=args.save_results, save_noise_info=True, save_stat=True, dataset_json=EXTERNAL_DATASET_JSON, clean_audio=False)


if __name__ == '__main__':
    main()
    # CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python3 predict.py --ckpt 87 --save_results false --unknown_clean_signal true
