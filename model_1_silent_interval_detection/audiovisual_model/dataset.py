import functools
import glob
import json
import math
import os
import random
import time
from itertools import groupby
from tqdm import tqdm

# python3 -m pip install --user imageio
import imageio
import librosa
import numpy as np
from pathlib import Path
import pylab
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from common import EXPERIMENT_NAME, PROJECT_ROOT, PHASE_TESTING, PHASE_TRAINING, PHASE_PREDICTION
from tools import *
from utils import ensure_dir, get_parent_dir, get_basename_no_ext
from visualization import draw_waveform, draw_spectrum
import cv2


DATA_ROOT = os.path.join(PROJECT_ROOT, "../", "data")
DEBUG_OUT = os.path.join(DATA_ROOT, "debug_dataset_output", EXPERIMENT_NAME)
NUM_DATA = 6000    # 100000
SILENT_CONSECUTIVE_FRAMES = 1
CLIP_FRAMES = 60
RANDOM_SEED = 10
PRED_RANDOM_SEED = 100
JSON_PARTIAL_NAME = '_TEDx1.json'

DATA_REQUIRED_SR = 14000
DATA_REQUIRED_FPS = 30.0
DATA_MAX_AUDIO_SAMPLES = int(math.floor(CLIP_FRAMES / DATA_REQUIRED_FPS * DATA_REQUIRED_SR))
NOISE_MAX_LENGTH_IN_SECOND = DATA_MAX_AUDIO_SAMPLES / DATA_REQUIRED_SR * 1.5  # times 1.5 to be safe

SNRS = [-10, -7, -3, 0, 3, 7, 10]

NOISE_SRC_ROOT_TRAIN = os.path.join(DATA_ROOT, "noise_data_DEMAND", "train_noise")
NOISE_SRC_ROOT_TEST = os.path.join(DATA_ROOT, "noise_data_DEMAND", "test_noise")

AUDIOSET_NOISE_SRC_TRAIN = os.path.join(DATA_ROOT, "audioset_noises_balanced_train")
AUDIOSET_NOISE_SRC_EVAL = os.path.join(DATA_ROOT, "audioset_noises_balanced_eval")


# Functions
##############################################################################
def get_dataloader(phase, batch_size=4, num_workers=4, snr_idx=None, dataset_json=None, clean_audio=True):
    print('Mode:', phase)

    num_data = NUM_DATA if phase == PHASE_TRAINING else NUM_DATA // 10
    is_shuffle = phase == PHASE_TRAINING

    # dataset
    dataset = AudioVisualAVSpeechMultipleVideoDataset(phase, num_data, CLIP_FRAMES, SILENT_CONSECUTIVE_FRAMES,\
        snr_idx, dataset_json=dataset_json, clean_audio=clean_audio)

    # dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle,\
        num_workers=num_workers, pin_memory=True, worker_init_fn=np.random.seed())

    return dataloader


# datasets
##############################################################################
class AudioVisualAVSpeechMultipleVideoDataset(Dataset):
    def __init__(self, phase, num_samples, clip_frames, consecutive_frames, snr_idx, dataset_json=None, clean_audio=True, n_fft=510, hop_length=158, win_length=400):
        print('========== DATASET CONSTRUCTION ==========')
        print('Initializing dataset...')
        super(AudioVisualAVSpeechMultipleVideoDataset, self).__init__()
        self.phase = phase
        if dataset_json is None:
            self.dataset_json = os.path.join(DATA_ROOT, phase + JSON_PARTIAL_NAME)
            # self.data_json = os.path.join(DATA_ROOT, phase + JSON_PARTIAL_NAME).split('.json')[0] + '_from_training.json'
        else:
            self.dataset_json = dataset_json
        self.clip_frames = clip_frames
        self.consecutive_frames = consecutive_frames
        self.clean_audio = clean_audio
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        print('Loading data...')
        with open(self.dataset_json, 'r') as fp:
            info = json.load(fp)
        self.dataset_path = info['dataset_path']
        self.num_files = info['num_videos']
        self.files = info['files']

        if clean_audio:
            # self.snrs = [0, 5, 10, 15]
            # if phase != PHASE_TRAINING:
            #     self.snrs = [2.5, 7.5, 12.5, 17.5]
            # self.snrs = [-20, -17, -13, -10, -7, -3, 0, 3, 7, 10]
            self.snrs = SNRS
            print("SNRs:", self.snrs)
            self.snr_idx = snr_idx
            print("snr_idx:", self.snr_idx)

            print('Getting all noise files...')
            # self.sr = 14000
            # self.fps = 30.0
            # self.max_audio_samples = int(math.floor(self.clip_frames / self.fps * self.sr))
            self.noise_src = [f.resolve() for f in Path(NOISE_SRC_ROOT_TRAIN).rglob('*.wav')]\
                + [f.resolve() for f in Path(AUDIOSET_NOISE_SRC_TRAIN).rglob('*.wav')]
            if phase != PHASE_TRAINING:
                self.noise_src = [f.resolve() for f in Path(NOISE_SRC_ROOT_TEST).rglob('*.wav')]\
                    + [f.resolve() for f in Path(AUDIOSET_NOISE_SRC_EVAL).rglob('*.wav')]
            # elif phase == PHASE_PREDICTION:
            #     self.noise_src = [f.resolve() for f in Path(NOISE_SRC_ROOT_TEST).rglob('*.wav')]
                # self.noise_src = sorted([f.resolve() for f in Path(get_parent_dir(self.dataset_json)).rglob('*_noise.wav')])
            # print(len(self.noise_src))
            # print(self.noise_src)

            print('Loading all noise files...')
            self.noises = [i[0] for i in (librosa.load(n, sr=DATA_REQUIRED_SR) for n in tqdm(self.noise_src))]
            # print(len(self.noises))
            self.noise_dict = {}
            if phase == PHASE_PREDICTION:
                random.seed(PRED_RANDOM_SEED)
                for f_idx, file in enumerate(self.files):
                    selected_noise = random.choice(self.noises)
                    # selected_noise = random_select_noises_for_pred(file, self.noises)
                    # selected_noise = self.noises[f_idx]
                    # print(os.path.basename(self.noise_src[f_idx]))
                    # print(os.path.basename(file['path']))
                    # # for now, noise and audio have to match
                    # assert get_basename_no_ext(file['path']) in get_basename_no_ext(self.noise_src[f_idx])

                    start = random.randint(0, len(selected_noise) - int(math.ceil(file['duration'])*DATA_REQUIRED_SR))
                    selected_noise_cropped = selected_noise[start:start+int(math.ceil(file['duration'])*DATA_REQUIRED_SR)]
                    # selected_noise_cropped = selected_noise
                    if self.snr_idx is None:
                        snr = random.choice(self.snrs)
                    else:
                        snr = self.snrs[self.snr_idx]
                    self.noise_dict[f_idx] = (selected_noise_cropped, snr)

        print('Generating data items...')
        # list of tuples (video_index, start_frame, bit_stream, center_label)
        # [(0, 450, '000011111111111', 1.0), ..., (79, 18349, '000111100000000', 0.0)]
        # self.items = create_sample_list_from_indices(self.files, num_samples, self.clip_frames)
        self.items = create_sample_list_from_indices(self.files, clip_frames=self.clip_frames,\
            silent_consecutive_frames=self.consecutive_frames, random_seed=RANDOM_SEED)
        if phase == PHASE_TESTING:
            self.items = create_sample_list_from_indices(self.files, num_samples=len(self.items)//10, clip_frames=self.clip_frames,\
                silent_consecutive_frames=self.consecutive_frames, random_seed=RANDOM_SEED)
        elif phase == PHASE_PREDICTION:
            self.items = create_sample_list_from_indices(self.files, clip_frames=self.clip_frames,\
                silent_consecutive_frames=self.consecutive_frames, random_seed=RANDOM_SEED, pred=True)
        # print(self.items)
        self.num_samples = len(self.items)
        # self.num_samples = num_samples

        print('========== SUMMARY ==========')
        print('Mode:', phase)
        print('Dataset JSON:', self.dataset_json)
        print('Dataset path:', self.dataset_path)
        print('Num samples:', self.num_samples)
        print('Data frames:', self.clip_frames)
        print('Consecutive frames:', self.consecutive_frames)
        print('Max noise length in seconds:', NOISE_MAX_LENGTH_IN_SECOND)
        print('Max audio samples per data:', DATA_MAX_AUDIO_SAMPLES)
        print('n_fft: {}'.format(self.n_fft))
        print('hop_length: {}'.format(self.hop_length))
        print('win_length: {}'.format(self.win_length))

    def __getitem__(self, index):
        item = self.items[index]
        # item[0]: video clip index
        # item[1]: data first bit's index in video clip
        # item[2]: data bit stream
        # item[3]: audio_path
        # item[4]: framerate
        file_info_dict = self.files[item[0]]
        # print(file_info_dict['path'])
        # file_info_dict['path']
        # file_info_dict['num_frames']
        # file_info_dict['bit_stream']
        assert item[1]+self.clip_frames <= int(file_info_dict['num_frames'])
        # print(index, item[1], file_info_dict['num_frames'])

        try:
            # Get labels
            labels = torch.tensor(item[2], dtype=torch.float32)
            # print('bit_stream:', item[2])

            # Get frames
            # frames = torch.stack(list(\
            #     map(functools.partial(load_image_from_index, dir_path=file_info_dict['frames_path']),\
            #         range(item[1], item[1]+self.clip_frames)))).permute(1, 0, 2, 3)

            # Data augmentation
            if self.phase == PHASE_TRAINING:
                # Get frames
                frames = torch.stack(list(\
                    map(functools.partial(load_image_from_index, dir_path=file_info_dict['frames_path'], randrot=random.uniform(-20, 20)),\
                        range(item[1], item[1]+self.clip_frames)))).permute(1, 0, 2, 3)
                frames = random_crop_batch(frames)
                frames = random_lrflip_batch(frames)
            elif self.phase == PHASE_TESTING:
                # Get frames
                frames = torch.stack(list(\
                    map(functools.partial(load_image_from_index, dir_path=file_info_dict['frames_path'], randrot=0),\
                        range(item[1], item[1]+self.clip_frames)))).permute(1, 0, 2, 3)
                frames = center_crop_batch(frames)
            else:
                # Get frames
                frames = torch.stack(list(\
                    map(functools.partial(load_image_from_index, dir_path=file_info_dict['frames_path'], randrot=0),\
                        range(item[1], item[1]+len(labels))))).permute(1, 0, 2, 3)
                frames = center_crop_batch(frames)

            # Get audio chunck
            snd, sr = librosa.load(item[3], sr=DATA_REQUIRED_SR)
            # snd = librosa.util.normalize(snd)
            # snd = audio_normalize(snd)

            # get corresponding audio chunk
            if self.phase != PHASE_PREDICTION:
                audio_raw = snd[int(item[1]/item[4]*sr):int((item[1]+self.clip_frames)/item[4]*sr)]
                audio = audio_raw[:DATA_MAX_AUDIO_SAMPLES]
                diff = DATA_MAX_AUDIO_SAMPLES - len(audio)
                if diff > 0:
                    audio = np.concatenate((audio, np.zeros(diff)))
            else:
                audio = snd
            # audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            # print('audio.shape1', audio.shape)

            # enforce silent intervals to be truly silent (clean_sig)
            # if self.phase != PHASE_PREDICTION:
            if self.clean_audio:
                mask = convert_bitstreammask_to_audiomask(audio, float(sr)/item[4], item[2])
                audio = audio * (1 - mask)
            # print('audio.shape2', audio.shape)

            # add noise to audio
            if self.clean_audio:
                if self.phase == PHASE_PREDICTION:
                    snr = self.noise_dict[item[0]][1]
                    noise = self.noise_dict[item[0]][0]
                    audio_mixed, audio_clean, _ = add_noise_to_audio(audio, noise, snr=snr, start_pos=int(item[1]/item[4]*sr), norm=0.5)
                    # raise NotImplementedError
                else:
                    if self.snr_idx is None:
                        snr = random.choice(self.snrs)
                    else:
                        snr = self.snrs[self.snr_idx]
                    noise = random.choice(self.noises)
                    # # start_time = time.time()
                    # noise_path, offset = random_select_data_as_noise(item, self.files, safe_len=NOISE_MAX_LENGTH_IN_SECOND)
                    # # print('noise_path:', noise_path)
                    # # print('offset:', offset)
                    # noise, _ = librosa.load(noise_path, sr=sr, duration=NOISE_MAX_LENGTH_IN_SECOND, offset=offset)
                    # # print('noise.shape', noise.shape)
                    # if len(noise) < DATA_MAX_AUDIO_SAMPLES:
                    #     print('len(noise):', len(noise))
                    #     print('len(noise) in sec:', len(noise)/sr)
                    #     print('desired len:', DATA_MAX_AUDIO_SAMPLES)
                    #     raise ValueError
                    # # print("--- %s seconds in getitem ---" % (time.time() - start_time))
                    audio_mixed, audio_clean, audio_noise = add_noise_to_audio(audio, noise, snr=snr, norm=0.5)

                # # audio_mixed = librosa.util.normalize(audio_mixed)
                # audio = torch.tensor(audio_mixed, dtype=torch.float32).unsqueeze(0)
                audio = audio_mixed
            else:
                # # audio_mixed = librosa.util.normalize(audio)
                # # scale = np.max(np.abs(audio_mixed)) / 0.5
                # # audio = torch.tensor(audio_mixed/scale, dtype=torch.float32).unsqueeze(0)
                # audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                pass

            # stft
            # print('audio.shape3', audio.shape)
            mixed_sig_stft = fast_stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)

            # # save images to visualize input
            # debug_out_more = os.path.join(DEBUG_OUT, str(snr))
            # ensure_dir(debug_out_more)

            # # print(file_info_dict)
            # print(snr)

            # # Debug: make silent data black
            # # frames = frames * 0;
            
            # # save_imgs_from_tensor(frames, debug_out_more)

            # # save audio
            # librosa.output.write_wav(os.path.join(debug_out_more, 'audio_mixed.wav'), audio_mixed, sr)
            # librosa.output.write_wav(os.path.join(debug_out_more, 'audio_clean.wav'), audio_clean, sr)
            # librosa.output.write_wav(os.path.join(debug_out_more, 'audio_noise.wav'), audio_noise[0], sr)
            
            # fig = plt.figure()
            # fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
            # ax1 = fig.add_subplot(2, 1, 1)
            # librosa.display.waveplot(audio_mixed, sr=sr, ax=ax1)
            # ax2 = fig.add_subplot(2, 1, 2)
            # s_spec = librosa.stft(audio_mixed, n_fft=510, hop_length=158, win_length=400)
            # librosa.display.specshow(librosa.amplitude_to_db(np.abs(s_spec), ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax2)
            # plt.savefig(os.path.join(debug_out_more, 'audio_mixed.jpg'))
            # plt.close(fig)

            # fig = plt.figure()
            # fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
            # ax1 = fig.add_subplot(2, 1, 1)
            # librosa.display.waveplot(audio_clean, sr=sr, ax=ax1)
            # ax2 = fig.add_subplot(2, 1, 2)
            # s_spec = librosa.stft(audio_clean, n_fft=510, hop_length=158, win_length=400)
            # librosa.display.specshow(librosa.amplitude_to_db(np.abs(s_spec), ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax2)
            # plt.savefig(os.path.join(debug_out_more, 'audio_clean.jpg'))
            # plt.close(fig)

            # fig = plt.figure()
            # fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
            # ax1 = fig.add_subplot(2, 1, 1)
            # librosa.display.waveplot(audio_noise[0], sr=sr, ax=ax1)
            # ax2 = fig.add_subplot(2, 1, 2)
            # s_spec = librosa.stft(audio_noise[0], n_fft=510, hop_length=158, win_length=400)
            # librosa.display.specshow(librosa.amplitude_to_db(np.abs(s_spec), ref=np.max), y_axis='log', x_axis='time', sr=sr, ax=ax2)
            # plt.savefig(os.path.join(debug_out_more, 'audio_noise.jpg'))
            # plt.close(fig)

            # exit(0)

        except Exception as e:
            # print(e)
            raise RuntimeError

        # print(labels)
        # print(label)
        # print(frames.shape)

        return {
            "frames": frames,
            "label": labels,
            "audio": mixed_sig_stft,
        }

    def __len__(self):
        return len(self.items)


def test():
    print('In test')
    # dataloader = get_dataloader(PHASE_TRAINING, batch_size=1, num_workers=1)
    dataloader = get_dataloader(PHASE_TESTING, batch_size=2, num_workers=0)
    # dataloader = get_dataloader(PHASE_PREDICTION, batch_size=1, num_workers=0)
    for i, data in enumerate(dataloader):
        print('================================================================')
        print('batch index:', i)
        print('data[\'frames\'].size():', data['frames'].size())
        print('data[\'label\'].size():', data['label'].size())
        print('data[\'label\']:', data['label'])
        print('data[\'audio\'].size():', data['audio'].size())
        print('min-max: ({}, {})'.format(torch.min(data['frames']).numpy().squeeze(),\
            torch.max(data['frames']).numpy().squeeze()))
        print('================================================================')

        # fig = plt.figure()
        # fig.subplots_adjust(left=0.2, right=0.95, hspace=0.3)
        # ax1 = fig.add_subplot(2, 1, 1)
        # librosa.display.waveplot(data['audio'].numpy().squeeze(), sr=DATA_REQUIRED_SR, ax=ax1)
        # ax2 = fig.add_subplot(2, 1, 2)
        # s_spec = librosa.stft(data['audio'].numpy().squeeze(), n_fft=510, hop_length=158, win_length=400)
        # librosa.display.specshow(librosa.amplitude_to_db(np.abs(s_spec), ref=np.max), y_axis='log', x_axis='time', sr=DATA_REQUIRED_SR, ax=ax2)
        # plt.savefig('audio.jpg')
        # plt.close(fig)
        # librosa.output.write_wav('audio.wav', data['audio'].numpy().squeeze(), DATA_REQUIRED_SR)
        # exit()

        if i >= 0:
            exit()


if __name__ == "__main__":
    test()
