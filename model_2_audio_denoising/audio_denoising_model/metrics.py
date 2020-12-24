import os
from collections import OrderedDict
import re
import numpy as np
import soundfile as sf
import tempfile
from subprocess import run, PIPE
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy.linalg import toeplitz
from pypesq import pesq
from pystoi.stoi import stoi
from itertools import groupby


def evaluate_metrics(noisy, clean, sr=16000, eps=1e-20):
    csig, cbak, covl, pesq_raw, ssnr, overall_snr = CompositeEval(clean, noisy, sr, eps=eps)
    metrics = OrderedDict()
    metrics['l1'] = metrics_L1(noisy, clean)
    metrics['stoi'] = metrics_stoi(noisy, clean, sr)
    metrics['csig'] = csig
    metrics['cbak'] = cbak
    metrics['covl'] = covl
    metrics['pesq'] = pesq_raw
    metrics['ssnr_regular'] = metrics_ssnr(clean, noisy, srate=sr, eps=eps)[1]
    metrics['ssnr_shift'] = metrics_ssnr_shift(clean, noisy, srate=sr, eps=eps)[1]
    # metrics['ssnr_clip'] = metrics_ssnr(clean, noisy, srate=sr, min_snr=0, eps=eps)[1]
    metrics['ssnr_clip'] = ssnr
    metrics['ssnr_exsi'] = metrics_ssnr_exclude_silence(clean, noisy, srate=sr, eps=eps)[1]
    metrics['overall_snr'] = overall_snr
    # print(metrics)
    return metrics


""" 
    Helper Functions
"""


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


def metrics_pesq_obsolete(ref_wav, deg_wav):
    # reference wav
    # degraded wav

    tfl = tempfile.NamedTemporaryFile()
    ref_tfl = tfl.name + '_ref.wav'
    deg_tfl = tfl.name + '_deg.wav'

    #if ref_wav.max() <= 1:
    #    ref_wav = np.array(denormalize_wave_minmax(ref_wav), dtype=np.int16)
    #if deg_wav.max() <= 1:
    #    deg_wav = np.array(denormalize_wave_minmax(deg_wav), dtype=np.int16)

    #wavfile.write(ref_tfl, 16000, ref_wav)
    #wavfile.write(deg_tfl, 16000, deg_wav)
    sf.write(ref_tfl, ref_wav, 16000, subtype='PCM_16')
    sf.write(deg_tfl, deg_wav, 16000, subtype='PCM_16')

    curr_dir = os.getcwd()
    # Write both to tmp files and then eval with pesqmain
    try:
        p = run(['pesqmain'.format(curr_dir), 
                 ref_tfl, deg_tfl, '+16000', '+wb'],
                stdout=PIPE, 
                encoding='ascii')
        res_line = p.stdout.split('\n')[-2]
        results = re.split('\s+', res_line)
        return results[-1]
    except FileNotFoundError:
        print('pesqmain not found! Please add it your PATH')


def metrics_ssnr(ref_wav, deg_wav, srate=16000, win_len=30, min_snr=-10, max_snr=35, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    # scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + eps))

    # global variables
    winlength = int(np.round(win_len * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    # MIN_SNR = -10
    # MAX_SNR = 35

    # For each frame, calculate SSNR

    num_frames = int(clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps) + eps))
        segmental_snr[-1] = max(segmental_snr[-1], min_snr)
        segmental_snr[-1] = min(segmental_snr[-1], max_snr)
        start += int(skiprate)
    return np.nanmean(overall_snr), np.nanmean(segmental_snr)


def metrics_ssnr_shift(ref_wav, deg_wav, srate=16000, win_len=30, min_snr=-10, max_snr=35, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    # scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + eps))

    # global variables
    winlength = int(np.round(win_len * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    # MIN_SNR = -10
    # MAX_SNR = 35

    # For each frame, calculate SSNR

    num_frames = int(clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps) + 1))
        segmental_snr[-1] = max(segmental_snr[-1], min_snr)
        segmental_snr[-1] = min(segmental_snr[-1], max_snr)
        start += int(skiprate)
    return np.nanmean(overall_snr), np.nanmean(segmental_snr)


def metrics_ssnr_exclude_silence(ref_wav, deg_wav, srate=16000, win_len=30, min_snr=-10, max_snr=35, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length
    # print('original length:', clean_length)

    # mask out silence
    new_clean_speech = []
    new_processed_speech = []
    start_idx = 0
    # for item in ((k, len(list(g))) for k, g in groupby(np.where(clean_speech == 0, 0, 1))):
    for item in ((k, len(list(g))) for k, g in groupby(np.where(np.abs(clean_speech) < np.max(np.abs(clean_speech))*0.03, 0, 1))):
        if item[0] == 1:
            # print(clean_speech[start_idx:start_idx+item[1]])
            new_clean_speech.append(clean_speech[start_idx:start_idx+item[1]])
            new_processed_speech.append(processed_speech[start_idx:start_idx+item[1]])
        start_idx += item[1]
    new_clean_speech = np.concatenate(new_clean_speech)
    new_processed_speech = np.concatenate(new_processed_speech)
    new_clean_length = new_clean_speech.shape[0]
    new_processed_length = new_processed_speech.shape[0]
    assert new_clean_length == new_processed_length
    # print('new length:', new_clean_length)
    # print(np.where(new_clean_speech == 0)[0])     # should return an empty array

    # scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + eps))

    # global variables
    winlength = int(np.round(win_len * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    # MIN_SNR = -10
    # MAX_SNR = 35

    # For each frame, calculate SSNR

    num_frames = int(new_clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = new_clean_speech[start:start+winlength]
        processed_frame = new_processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps) + eps))
        segmental_snr[-1] = max(segmental_snr[-1], min_snr)
        segmental_snr[-1] = min(segmental_snr[-1], max_snr)
        start += int(skiprate)
    # return np.nanmean(overall_snr), np.nanmean(segmental_snr)
    return np.nanmean(overall_snr), np.nanmean(segmental_snr)


def metrics_ssnr_exclude_silence_shift(ref_wav, deg_wav, srate=16000, win_len=30, min_snr=-10, max_snr=35, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length
    # print('original length:', clean_length)

    # mask out silence
    new_clean_speech = []
    new_processed_speech = []
    start_idx = 0
    # for item in ((k, len(list(g))) for k, g in groupby(np.where(clean_speech == 0, 0, 1))):
    for item in ((k, len(list(g))) for k, g in groupby(np.where(np.abs(clean_speech) < np.max(np.abs(clean_speech))*0.03, 0, 1))):
        if item[0] == 1:
            # print(clean_speech[start_idx:start_idx+item[1]])
            new_clean_speech.append(clean_speech[start_idx:start_idx+item[1]])
            new_processed_speech.append(processed_speech[start_idx:start_idx+item[1]])
        start_idx += item[1]
    new_clean_speech = np.concatenate(new_clean_speech)
    new_processed_speech = np.concatenate(new_processed_speech)
    new_clean_length = new_clean_speech.shape[0]
    new_processed_length = new_processed_speech.shape[0]
    assert new_clean_length == new_processed_length
    # print('new length:', new_clean_length)
    # print(np.where(new_clean_speech == 0)[0])     # should return an empty array

    # scale both to have same dynamic range. Remove DC too.
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) + eps))

    # global variables
    winlength = int(np.round(win_len * srate / 1000)) # 30 msecs
    skiprate = winlength // 4
    # MIN_SNR = -10
    # MAX_SNR = 35

    # For each frame, calculate SSNR

    num_frames = int(new_clean_length / skiprate - (winlength/skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = new_clean_speech[start:start+winlength]
        processed_frame = new_processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps) + 1))
        segmental_snr[-1] = max(segmental_snr[-1], min_snr)
        segmental_snr[-1] = min(segmental_snr[-1], max_snr)
        start += int(skiprate)
    return np.nanmean(overall_snr), np.nanmean(segmental_snr)


def metrics_ssnr_rundi(output, target, sr=16000, frame_len=20, min_snr=-10, max_snr=35, eps=1e-10):
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
    value = 10 * np.log10(np.sum(target_segs ** 2, axis=1) / (np.sum((target_segs - output_segs) ** 2, axis=1) + eps) + eps)
    result += np.sum(np.clip(value, min_snr, max_snr), axis=0)

    # remaining tail
    value = 10 * np.log(np.sum(target[-remains:] ** 2) / np.sum((target[-remains:] - output[-remains:]) ** 2 + eps) + eps)
    result += np.clip(value, min_snr, max_snr)

    result /= (n_segs + 1)
    return result


def metrics_stoi(output, target, sr=16000, extended=False):
    # stoi
    return stoi(target, output, sr, extended=extended)


def CompositeEval(ref_wav, deg_wav, srate=16000, eps=1e-10):
    # returns [sig, bak, ovl]
    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    ref_len = ref_wav.shape[0]
    deg_wav = deg_wav[:len_]

    # Compute WSS measure
    wss_dist_vec = wss(ref_wav, deg_wav, srate, eps=eps)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.nanmean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])
    # print('wss_dist:', wss_dist)

    # Compute LLR measure
    LLR_dist = llr(ref_wav, deg_wav, srate)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.nanmean(LLRs[:LLR_len])
    # print('llr_mean:', llr_mean)

    # Compute the SSNR
    # snr_mean, segsnr_mean = metrics_ssnr(ref_wav, deg_wav, win_len=30, srate=srate, eps=eps)
    # segSNR = np.nanmean(segsnr_mean)
    # overall_snr = np.nanmean(snr_mean)
    # overall_snr, segSNR = metrics_ssnr(ref_wav, deg_wav, srate=srate, eps=eps)
    overall_snr, segSNR = metrics_ssnr(ref_wav, deg_wav, srate=srate, min_snr=0, eps=eps)
    # print('segSNR:', segSNR)

    # Compute the PESQ
    # pesq_raw = PESQ(ref_wav, deg_wav)
    pesq_raw = metrics_pesq(deg_wav, ref_wav, srate)
    # if 'error!' not in pesq_raw:
    #     pesq_raw = float(pesq_raw)
    # else:
    #     pesq_raw = -1.
    # print('pesq_raw:', pesq_raw)

    def trim_mos(val):
        return min(max(val, 1), 5)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    # print('Csig 1:', Csig)
    Csig = trim_mos(Csig)
    # print('Csig 2:', Csig)
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    # print('Cbak 1:', Cbak)
    Cbak = trim_mos(Cbak)
    # print('Cbak 2:', Cbak)
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    # print('Covl 1:', Covl)
    Covl = trim_mos(Covl)
    # print('Covl 2:', Covl)

    return Csig, Cbak, Covl, pesq_raw, segSNR, overall_snr


def wss(ref_wav, deg_wav, srate, eps=1e-10):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25 # num of critical bands

    USE_FFT_SPECTRUM = 1
    n_fft = int(2 ** np.ceil(np.log(2*winlength)/np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1

    # Critical band filter definitions (Center frequency and BW in Hz)

    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30, 
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                 95.3398, 105.411, 116.256, 127.914, 140.423, 
                 153.823, 168.154, 183.457, 199.776, 217.153, 
                 235.631, 255.255, 276.072, 298.126, 321.465,
                 346.136]

    bw_min = bandwidth[0] # min critical bandwidth

    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.

    min_factor = np.exp(-30. / (2 * 2.303)) # -30 dB point of filter

    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0.append(np.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + \
                                   norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > \
                                                 min_factor)
    # For each frame of input speech, compute Weighted Spectral Slope Measure

    # num of frames
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0 # starting sample
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):

        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compuet Power Spectrum of clean and processed

        clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
        processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit
        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * \
                                     crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * \
                                         crit_filter[i, :])
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps_np = np.ones((clean_energy.shape[0], 1)) * eps
        clean_energy = np.concatenate((clean_energy, eps_np), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps_np), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))
        # (4) Compute Spectral Shape (dB[i+1] - dB[i])

        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit-1]
        processed_slope = processed_energy[1:num_crit] - \
                processed_energy[:num_crit-1]
        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])
        # (6) Compuet the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)
        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral 
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - \
                                   clean_energy[:num_crit-1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - \
                                processed_energy[:num_crit-1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - \
                                      processed_energy[:num_crit-1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - \
                                     processed_slope[:num_crit - 1]) ** 2))

        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion


def llr(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.) # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio

    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):

        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start+winlength]
        processed_frame = processed_speech[start:start+winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window
        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]
        #print('A_clean shape: ', A_clean.shape)
        #print('toe(R_clean) shape: ', toeplitz(R_clean).shape)
        #print('A_clean: ', A_clean)
        #print('A_processed: ', A_processed)
        #print('toe(R_clean): ', toeplitz(R_clean))
        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        #print('num_1: {}'.format(A_processed.dot(toeplitz(R_clean))))
        #print('num: ', numerator)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)
        #print('den: ', denominator)
        #log_ = np.log(max(numerator / denominator, 10e-20))
        #print('R_clean: ', R_clean)
        # print('num: ', numerator)
        # print('den: ', denominator)
        #raise NotImplementedError
        # if numerator / denominator < 0:
        #     print('negative', numerator / denominator)
        # if numerator / denominator == 0:
        #     print('zero', numerator / denominator)
        log_ = np.log(numerator / denominator)
        #print('np.log({}/{}) = {}'.format(numerator, denominator, log_))
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.array(distortion)

#@nb.jit('UniTuple(float32[:], 3)(float32[:])')#,nopython=True)
def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocor lags
    # max?
    winlength = speech_frame.shape[0]
    R = []
    #R = [0] * (model_order + 1)
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        #raise NotImplementedError
        R.append(np.sum(first * second))
        #R[k] = np.sum( first * second)
    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        #print('-' * 40)
        #print('i: ', i)
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            #print('R[i:0:-1] = ', R[i:0:-1])
            #print('a_past = ', a_past)
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
            #print('a_past size: ', a_past.shape)
        #print('sum_term = {:.6f}'.format(sum_term))
        #print('E[i] =  {}'.format(E[i]))
        #print('R[i+1] = ', R[i+1])
        rcoeff[i] = (R[i+1] - sum_term)/E[i]
        #print('len(a) = ', len(a))
        #print('len(rcoeff) = ', len(rcoeff))
        #print('a[{}]={}'.format(i, a[i]))
        #print('rcoeff[{}]={}'.format(i, rcoeff[i]))
        a[i] = rcoeff[i]
        if i > 0:
            #print('a: ', a)
            #print('a_past: ', a_past)
            #print('a_past[:i] ', a_past[:i])
            #print('a_past[::-1] ', a_past[::-1])
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i+1] = (1-rcoeff[i]*rcoeff[i])*E[i]
        #print('E[i+1]= ', E[i+1])
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr =np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)
    #print('acorr shape: ', acorr.shape)
    #print('refcoeff shape: ', refcoeff.shape)
    #print('lpparams shape: ', lpparams.shape)
    return acorr, refcoeff, lpparams
