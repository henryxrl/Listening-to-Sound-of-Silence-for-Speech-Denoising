import glob
import json
import os
import random
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (EXPERIMENT_NAME, PHASE_PREDICTION, PHASE_TESTING,
                    PHASE_TRAINING, PROJECT_ROOT)
from tools import *
from transform import *
from utils import ensure_dir
from visualization import draw_spectrum, draw_waveform


DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DEBUG_OUT = os.path.join(DATA_ROOT, "debug_dataset_output", EXPERIMENT_NAME)
JSON_PARTIAL_NAME = '_TEDx3.json'

RANDOM_SEED = 10
PRED_RANDOM_SEED = 100

DATA_LENGTH_SECONDS = 2
DATA_OVERLAP_SECONDS = 1
DATA_REQUIRED_SR = 14000

NOISE_SRC_ROOT_TRAIN = os.path.join(DATA_ROOT, "noise_data_DEMAND", "train_noise")
NOISE_SRC_ROOT_TEST = os.path.join(DATA_ROOT, "noise_data_DEMAND", "test_noise")

AUDIOSET_NOISE_SRC_TRAIN = os.path.join(DATA_ROOT, "audioset_noises_balanced_train")
AUDIOSET_NOISE_SRC_EVAL = os.path.join(DATA_ROOT, "audioset_noises_balanced_eval")

# Functions
##############################################################################
def get_dataloader(phase, batch_size=4, num_workers=4, snr_idx=None):
    is_shuffle = phase == PHASE_TRAINING

    dataset = AudioDataset(phase, DATA_LENGTH_SECONDS, DATA_OVERLAP_SECONDS, sr=DATA_REQUIRED_SR, snr_idx=snr_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers,
                            pin_memory=True, worker_init_fn=np.random.seed())
    return dataloader


# datasets
##############################################################################
class AudioDataset(Dataset):
    def __init__(self, phase, data_len_sec, data_overlap_sec, sr=16000, n_fft=510, hop_length=158, win_length=400, snr_idx=None):
        print('========== DATASET CONSTRUCTION ==========')
        print('Initializing dataset...')
        super(AudioDataset, self).__init__()
        self.data_root = os.path.join(DATA_ROOT, phase)
        self.aug = phase == PHASE_TRAINING
        self.phase = phase
        self.sr = sr
        self.data_len_sec = data_len_sec
        self.data_overlap_sec = data_overlap_sec
        # self.clip_len = 32768 # 28000
        self.clip_len = self.sr * data_len_sec
        self.clip_overlap = self.sr * data_overlap_sec
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.dataset_json = os.path.join(DATA_ROOT, phase + JSON_PARTIAL_NAME)

        print('Loading data...')
        with open(os.path.join(self.dataset_json), 'r') as fp:
            info = json.load(fp)
        self.dataset_path = info['dataset_path']
        self.num_files = info['num_videos']
        self.files = info['files']
        # self.items = info[phase]

        print('Getting all noise files...')
        self.noise_src = [f.resolve() for f in Path(NOISE_SRC_ROOT_TRAIN).rglob('*.wav')]\
            + [f.resolve() for f in Path(AUDIOSET_NOISE_SRC_TRAIN).rglob('*.wav')]
        if phase != PHASE_TRAINING:
            self.noise_src = [f.resolve() for f in Path(NOISE_SRC_ROOT_TEST).rglob('*.wav')]\
                + [f.resolve() for f in Path(AUDIOSET_NOISE_SRC_EVAL).rglob('*.wav')]
        # print(len(self.noise_src))

        # self.snrs = [0, 5, 10, 15]
        # if phase != PHASE_TRAINING:
        #     self.snrs = [2.5, 7.5, 12.5, 17.5]
        # self.snrs = [-20, -17, -13, -10, -7, -3, 0, 3, 7, 10]
        self.snrs = [-10, -7, -3, 0, 3, 7, 10]
        # self.snrs = [0]    # train and test on single SNR
        print("SNRs:", self.snrs)
        self.snr_idx = snr_idx
        print("snr_idx:", self.snr_idx)

        print('Loading all noise files...')
        self.noises = [i[0] for i in (librosa.load(n, sr=self.sr) for n in tqdm(self.noise_src))]
        # print(len(self.noises))
        self.noise_dict = {}
        if phase == PHASE_PREDICTION:
            random.seed(PRED_RANDOM_SEED)
            for f_idx, file in enumerate(self.files):
                selected_noise = random.choice(self.noises)
                start = random.randint(0, len(selected_noise) - int(math.ceil(file['duration'])*self.sr))
                selected_noise_cropped = selected_noise[start:start+int(math.ceil(file['duration'])*self.sr)]
                if self.snr_idx is None:
                    snr = random.choice(self.snrs)
                else:
                    snr = self.snrs[self.snr_idx]
                self.noise_dict[f_idx] = (selected_noise_cropped, snr)

        print('Generating data items...')
        self.items = []
        if phase == PHASE_TRAINING:
            self.items = create_sample_list_from_indices(self.files, data_len_sec=self.data_len_sec,\
                data_overlap_sec=self.data_overlap_sec, random_seed=RANDOM_SEED)
        if phase == PHASE_TESTING:
            self.items = create_sample_list_from_indices(self.files, percent_samples_selected=0.1, data_len_sec=self.data_len_sec,\
                data_overlap_sec=self.data_overlap_sec, random_seed=RANDOM_SEED)
        elif phase == PHASE_PREDICTION:
            self.items = create_sample_list_from_indices(self.files, data_len_sec=self.data_len_sec,\
                data_overlap_sec=self.data_overlap_sec, random_seed=RANDOM_SEED, pred=True)
        # print(self.items)
        self.num_samples = len(self.items)
        # self.num_samples = num_samples

        print('========== SUMMARY ==========')
        print('Mode:', phase)
        print('Dataset JSON:', self.dataset_json)
        print('Dataset path:', self.dataset_path)
        print('Num samples:', self.num_samples)
        print('Sample rate: {}'.format(self.sr))
        print('Clip length: {}'.format(self.clip_len))
        print('n_fft: {}'.format(self.n_fft))
        print('hop_length: {}'.format(self.hop_length))
        print('win_length: {}'.format(self.win_length))

    def __getitem__(self, index):
        item = self.items[index]
        # item[0]: video clip index
        # item[1]: data starting bit's index (in second) in video clip
        # item[2]: data ending bit's index (in second) in video clip
        # item[3]: data bit stream
        # item[4]: audio_path
        # item[5]: framerate
        file_info_dict = self.files[item[0]]
        # print(item[1]+self.data_len_sec, '<=', float(file_info_dict['duration']))
        assert item[1]+self.data_len_sec <= float(file_info_dict['duration'])
        start = int(item[1] * self.sr)
        end = int(item[2] * self.sr)

        try:
            # TODO: read data
            # 1. read clean signal
            # clean signal
            audio, _ = librosa.load(item[4], sr=self.sr)
            audio = audio[start:end]
            # print('audio: ({}, {})'.format(np.amin(audio), np.amax(audio)))

            # 5. ground truth bitstream -> mask
            bitstream = item[3]
            frames_to_audiosample_ratio = self.sr / item[5]
            # print(mixed_sig.shape[0], '==', len(bitstream) * frames_to_audiosample_ratio)
            # assert mixed_sig.shape[0] == int(len(bitstream) * frames_to_audiosample_ratio)
            mask = np.zeros_like(audio)
            for bit_idx, bit in enumerate(bitstream):
                # mask out non-silent intervals in mixed_sig
                # silent 1. non-silent 0
                if bit == '0':    # silent frame
                    mask[int(bit_idx * frames_to_audiosample_ratio):int((bit_idx+1) * frames_to_audiosample_ratio - 1)] = 1
                elif bit == '1':  # non-silent frame
                    mask[int(bit_idx * frames_to_audiosample_ratio):int((bit_idx+1) * frames_to_audiosample_ratio - 1)] = 0
                else:
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

            # 5.5. enforce silent intervals to be truly silent (clean_sig)
            audio = audio * (1 - mask)

            # 2. read noise signal
            if self.phase == PHASE_PREDICTION:
                snr = self.noise_dict[item[0]][1]
                noise = self.noise_dict[item[0]][0]
            else:
                if self.snr_idx is None:
                    snr = random.choice(self.snrs)
                else:
                    snr = self.snrs[self.snr_idx]
                noise = random.choice(self.noises)

            # print('noise: ({}, {})'.format(np.amin(noise), np.amax(noise)))

            # 3. normalize clean/noise signal to the same range
            # audio = librosa.util.normalize(audio)
            # noise = librosa.util.normalize(noise)

            # print('audio normalized: ({}, {})'.format(np.amin(audio), np.amax(audio)))
            # print('noise normalized: ({}, {})'.format(np.amin(noise), np.amax(noise)))

            # 4. mixed_sig, clean_sig, full_noise = add_signals(clean, noise, snr=?, norm=0.5)
            # mixed_sig = clean_sig + full_noise
            mixed_sig, clean_sig, full_noises = add_noise_to_audio(audio, noise, snr=snr, norm=0.5)
            full_noise = full_noises[0]
            # print(mixed_sig.shape, clean_sig.shape, full_noise.shape)
            # assert mixed_sig.shape == clean_sig.shape == full_noise.shape

            # print('clean_sig: ({}, {})'.format(np.amin(clean_sig), np.amax(clean_sig)))
            # print('mixed_sig: ({}, {})'.format(np.amin(mixed_sig), np.amax(mixed_sig)))
            # print('full_noise: ({}, {})'.format(np.amin(full_noise), np.amax(full_noise)))
            # print('mixed == clean + full_noise?', np.allclose(mixed_sig, clean_sig+full_noise))

            
            # 6. noise_sig(masked noise) = full_noise * mask = mixed_sig * mask     # not really...
            noise_sig = mixed_sig * mask
            # assert np.alltrue(full_noise * mask == mixed_sig * mask)
            # print('noise_sig: ({}, {})'.format(np.amin(noise_sig), np.amax(noise_sig)))

            # stft
            mixed_sig_stft = fast_stft(mixed_sig, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            clean_sig_stft = fast_stft(clean_sig, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            noise_sig_stft = fast_stft(noise_sig, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            full_noise_sig_stft = fast_stft(full_noise, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

            # mixed_sig_stft = librosa.stft(mixed_sig, self.n_fft, self.hop_length, self.win_length)
            # clean_sig_stft = librosa.stft(clean_sig, self.n_fft, self.hop_length, self.win_length)
            # noise_sig_stft = librosa.stft(noise_sig, self.n_fft, self.hop_length, self.win_length)
            # full_noise_sig_stft = librosa.stft(full_noise, self.n_fft, self.hop_length, self.win_length)
            # mixed_sig_stft = real_imag_expand(mixed_sig_stft)
            # clean_sig_stft = real_imag_expand(clean_sig_stft)
            # noise_sig_stft = real_imag_expand(noise_sig_stft)
            # full_noise_sig_stft = real_imag_expand(full_noise_sig_stft)

            icrm = fast_cRM_sigmoid(clean_sig_stft, mixed_sig_stft)

            # # scale to [-1, 1]
            # mixed_sig = torch.tensor(mixed_sig, dtype=torch.float32).unsqueeze(0) / 0.5
            # clean_sig = torch.tensor(clean_sig, dtype=torch.float32).unsqueeze(0) / 0.5
            # noise_sig = torch.tensor(noise_sig, dtype=torch.float32).unsqueeze(0) / 0.5

            mixed_sig_stft = torch.tensor(mixed_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
            clean_sig_stft = torch.tensor(clean_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
            noise_sig_stft = torch.tensor(noise_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
            full_noise_sig_stft = torch.tensor(full_noise_sig_stft.transpose((2, 0, 1)), dtype=torch.float32)
            icrm = torch.tensor(icrm.transpose((2, 0, 1)), dtype=torch.float32)

            # For debug, visualizing data
            # debug_out_more = os.path.join(DEBUG_OUT, str(snr))
            # ensure_dir(debug_out_more)

            # # plt.figure()
            # # librosa.display.waveplot(mixed_sig, sr=self.sr)
            # # plt.savefig(os.path.join(debug_out_more, 'mixed_sig.jpg'))
            # # plt.close()
            # cv2.imwrite(os.path.join(debug_out_more, 'mixed_sig.jpg'), draw_waveform([mixed_sig], sr=self.sr))
            # cv2.imwrite(os.path.join(debug_out_more, 'mixed_sig_spec.jpg'), draw_spectrum([mixed_sig], sr=self.sr))
            # librosa.output.write_wav(os.path.join(debug_out_more, 'mixed_sig.wav'), mixed_sig, self.sr)

            # # plt.figure()
            # # librosa.display.waveplot(clean_sig, sr=self.sr)
            # # plt.savefig(os.path.join(debug_out_more, 'clean_sig.jpg'))
            # # plt.close()
            # cv2.imwrite(os.path.join(debug_out_more, 'clean_sig.jpg'), draw_waveform([clean_sig], sr=self.sr))
            # cv2.imwrite(os.path.join(debug_out_more, 'clean_sig_spec.jpg'), draw_spectrum([clean_sig], sr=self.sr))
            # librosa.output.write_wav(os.path.join(debug_out_more, 'clean_sig.wav'), clean_sig, self.sr)

            # # plt.figure()
            # # librosa.display.waveplot(full_noise, sr=self.sr)
            # # plt.savefig(os.path.join(debug_out_more, 'full_noise.jpg'))
            # # plt.close()
            # cv2.imwrite(os.path.join(debug_out_more, 'full_noise.jpg'), draw_waveform([full_noise], sr=self.sr))
            # cv2.imwrite(os.path.join(debug_out_more, 'full_noise_spec.jpg'), draw_spectrum([full_noise], sr=self.sr))
            # librosa.output.write_wav(os.path.join(debug_out_more, 'full_noise.wav'), full_noise, self.sr)

            # # plt.figure()
            # # librosa.display.waveplot(full_noise * mask, sr=self.sr)
            # # plt.savefig(os.path.join(debug_out_more, 'masked_noise_fullnoise.jpg'))
            # # plt.close()
            # cv2.imwrite(os.path.join(debug_out_more, 'masked_noise_fullnoise.jpg'), draw_waveform([full_noise * mask], sr=self.sr))
            # cv2.imwrite(os.path.join(debug_out_more, 'masked_noise_fullnoise_spec.jpg'), draw_spectrum([full_noise * mask], sr=self.sr))
            # librosa.output.write_wav(os.path.join(debug_out_more, 'masked_noise_fullnoise.wav'), full_noise * mask, self.sr)

            # # plt.figure()
            # # librosa.display.waveplot(mixed_sig * mask, sr=self.sr)
            # # plt.savefig(os.path.join(debug_out_more, 'masked_noise_mixedsig.jpg'))
            # # plt.close()
            # cv2.imwrite(os.path.join(debug_out_more, 'masked_noise_mixedsig.jpg'), draw_waveform([mixed_sig * mask], sr=self.sr))
            # cv2.imwrite(os.path.join(debug_out_more, 'masked_noise_mixedsig_spec.jpg'), draw_spectrum([mixed_sig * mask], sr=self.sr))
            # librosa.output.write_wav(os.path.join(debug_out_more, 'masked_noise_mixedsig.wav'), mixed_sig * mask, self.sr)

            # exit(0)

        except Exception as e:
            # print(e)
            raise RuntimeError

        return {
            "mixed": mixed_sig_stft,
            "clean": clean_sig_stft,
            "noise": noise_sig_stft,
            "full_noise": full_noise_sig_stft,
            "mask": icrm,
            # "id": item[0],
            "start": start,
            "bitstream": bitstream
        }

    def __len__(self):
        return len(self.items)


def test():
    # dataloader = get_dataloader(PHASE_TRAINING, batch_size=8, num_workers=0)
    dataloader = get_dataloader(PHASE_TESTING, batch_size=8, num_workers=10)
    # dataloader = get_dataloader(PHASE_PREDICTION, batch_size=1, num_workers=0)
    for i, data in enumerate(dataloader):
        print('================================================================')
        print('batch index:', i)
        print('data[\'bitstream\']:', data['bitstream'])
        print('data[\'mixed\'].shape:', data['mixed'].shape)
        print('data[\'clean\'].shape:', data['clean'].shape)
        print('data[\'noise\'].shape:', data['noise'].shape)
        print('data[\'mask\'].shape:', data['mask'].shape)
        # print(torch.max(data['mask']), torch.min(data['mask']))
        print('min-max: ({}, {})'.format(torch.min(data['mask']).numpy().squeeze(),\
            torch.max(data['mask']).numpy().squeeze()))
        print(data['mask'][0])

        # mixed = data['mixed'].numpy()
        # clean = data['clean'].numpy()
        # mask = data['mask'].numpy()
        # out = fast_icRM(mixed, mask, K=1)
        # print(clean == out)

        print('================================================================')
        if i >= 10:
            exit()

        if torch.max(data['mixed']) > 1 + 1e-8:
            print(torch.max(data['mixed']), torch.min(data['mixed']))
            exit()
        pass


if __name__ == "__main__":
    test()
