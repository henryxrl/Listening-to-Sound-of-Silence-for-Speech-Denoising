import gc
import glob
import json
import math
import os
import pprint
import re
import shutil
import subprocess
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
from pytube import YouTube
from scipy import ndimage

# from flow_extractor import convert_flow, extract_flow
# from face_extractor import extract_all_faces, get_initial_bb, get_time_series
from tools import *

mpl.use('Agg')


def youtube_dl(youtube_id, start_time, end_time, path, fr=29.97, suppress_stdout=True):
    """Download a Youtube video given id and range; change framerate if needed"""
    to_return = True   # download success!

    if not os.path.exists(path):
        try:
            sys_run_cmd('ffmpeg -loglevel fatal -i $(youtube-dl -4 -f 22 -g \'https://www.youtube.com/watch?v={}\')\
                -ss {} -to {} -qscale 0 -r {} -y "{}"'.format(\
                        youtube_id, start_time, end_time, fr, path))
            to_return = os.path.exists(path)
        except:
            to_return = False

    if to_return:
        if not suppress_stdout:
            print('Duration: {} seconds'.format(get_duration2(path)))
            print('Framerate: {}'.format(get_framerate(path)))
            print('Num frames: {}'.format(get_numframes(path)))
            print('Audio info:', get_samplerate_numaudiosamples(path))

    return to_return


def youtube_dl2(youtube_id, time_intervals, output_dir, filename, fr=29.97, suppress_stdout=True):
    """Download a Youtube video given id and time intervals; change framerate if needed"""
    to_return = True   # download success!

    # path = os.path.join(output_dir, '{}_{:07}{}'.format(filename, count+1, VIDEO_EXT))
    path = os.path.join(output_dir, filename + VIDEO_EXT)

    try:
        youtube_dl_full(youtube_id, path, fr, suppress_stdout)
        to_return = os.path.exists(path)
    except:
        to_return = False

    if to_return:
        for i, t in enumerate(time_intervals):
            out_path = os.path.join(output_dir, '{}_{:07}{}'.format(filename, i+1, VIDEO_EXT))
            sys_run_cmd('ffmpeg -loglevel fatal -i {} -ss {} -to {} -qscale 0 -r {} -y "{}"'\
                .format(path, t[0], t[1], fr, out_path))
            to_return *= os.path.exists(out_path)

        if not suppress_stdout:
            print('Duration: {} seconds'.format(get_duration2(path)))
            print('Framerate: {}'.format(get_framerate(path)))
            print('Num frames: {}'.format(get_numframes(path)))
            print('Audio info:', get_samplerate_numaudiosamples(path))

    return to_return


def youtube_dl_full(youtube_id, path, fr=29.97, suppress_stdout=True, method=0):
    """Download a Youtube video given id and range; change framerate if needed"""
    to_return = True   # download success!

    if not os.path.exists(path):
        try:
            if method == 0:
                sys_run_cmd('ffmpeg -loglevel fatal -i\
                    $(youtube-dl -4 -f 22 -g \'https://www.youtube.com/watch?v={}\')\
                        -qscale 0 -r {} -y "{}"'.format(youtube_id, fr, path))
                to_return = os.path.exists(path)
            elif method == 1:
                # sys_run_cmd('ffmpeg -loglevel fatal -i\
                #     $(youtube-dl -f 22 -g \'https://www.youtube.com/watch?v={}\')\
                #         -qscale 0 -r {} -y "{}"'.format(youtube_id, fr, path))

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_video_path = os.path.join(temp_dir, 'test'+VIDEO_EXT)
                    yt = YouTube('\'https://www.youtube.com/watch?v={}\''.format(youtube_id))
                    # yt = YouTube('https://www.youtube.com/watch?v=5b5BDoddOLA')
                    yt.streams.filter(progressive=True, file_extension=VIDEO_EXT.split('.')[1])\
                        .order_by('resolution').desc().first().download(temp_video_path)

                    sys_run_cmd('ffmpeg -loglevel fatal -i "{}"\
                        -qscale 0 -r {} -y "{}"'.format(temp_video_path, fr, path))

                to_return = os.path.exists(path)
        except Exception as e:
            print(e)
            to_return = False

    if to_return:
        if not suppress_stdout:
            print('Duration: {} seconds'.format(get_duration2(path)))
            print('Framerate: {}'.format(get_framerate(path)))
            print('Num frames: {}'.format(get_numframes(path)))
            print('Audio info:', get_samplerate_numaudiosamples(path))

    return to_return


def cut_video(input_path, start_time, end_time, output_path, fr=29.97, suppress_stdout=True):
    to_return = True   # download success!

    if not os.path.exists(output_path):
        try:
            sys_run_cmd('ffmpeg -loglevel fatal -i {} -ss {} -to {} -qscale 0 -r {} -y "{}"'\
                    .format(input_path, start_time, end_time, fr, output_path))
            # print(output_path)
            to_return = os.path.exists(output_path)
        except Exception as e:
            print(str(e))
            to_return = False

    # print(to_return)

    if to_return:
        if not suppress_stdout:
            print('Duration: {} seconds'.format(get_duration2(output_path)))
            print('Framerate: {}'.format(get_framerate(output_path)))
            print('Num frames: {}'.format(get_numframes(output_path)))
            print('Audio info:', get_samplerate_numaudiosamples(output_path))

    return to_return


def is_TED(youtube_id):
    """Check whether a Youtube video is a TED talk video"""
    to_return = False

    try:
        json_obj = json.loads(sys_run_cmd_with_stdout(\
            'youtube-dl --dump-json \'https://www.youtube.com/watch?v={}\''.format(youtube_id)))
        uploader_id = json_obj['uploader_id']
        # print(uploader_id)
        to_return = "TEDtalksDirector" in uploader_id
    except Exception as e:
        to_return = False

    return to_return


def is_TED2(youtube_id):
    """Check whether a Youtube video is a TED talk video"""
    to_return = ""

    try:
        json_obj = json.loads(sys_run_cmd_with_stdout(\
            'youtube-dl -4 --dump-json \'https://www.youtube.com/watch?v={}\''.format(youtube_id)))
        uploader_id = json_obj['uploader_id']
        # print(uploader_id)
        to_return = youtube_id if "TEDtalksDirector" in uploader_id else ""
    except Exception as e:
        to_return = ""

    return to_return


def TED_check_save(youtube_id, output_json):
    result = is_TED2(youtube_id)
    if result:
        with open(output_json, 'a') as f:
            f.write(result+'\n')
    return result


def get_framerate(path):
    """Get the framerate of a video"""
    output = sys_run_cmd_with_stdout('ffmpeg -i "{}" 2>&1'.format(path))
    lines = list(grep_str(output, ': Video: '))
    assert len(lines)
    m = re.match(r".* (\d+\.?\d*) tbr", lines[0])
    return None if (m is None) else float(m.group(1))


def change_framerate(path, fr=29.97, ext=VIDEO_EXT):
    """Change the framerate of a video"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video_path = os.path.join(temp_dir, os.path.basename(path))
        sys_run_cmd(
            # 'ffmpeg -loglevel fatal -y -r {} -i "{}" "{}"'.format(fr, path, temp_video_path))
            'ffmpeg -loglevel fatal -i "{}" -qscale 0 -r {} -y "{}"'.format(\
                path, fr, temp_video_path))
        if get_framerate(temp_video_path) != fr:
            print("Failed to change framerate!")
            return 2
        d1 = get_duration(path)
        d2 = get_duration(temp_video_path)
        print('Original duration: {} seconds'.format(d1))
        print('Converted duration: {} seconds'.format(d2))
        if isclose(d1, d2):
            shutil.move(temp_video_path, path)
            print('Success!')
            return 0
        else:
            path_new = path.split(ext)[0] + '_new' + ext
            print('Create a new file {}'.format(path_new))
            shutil.move(temp_video_path, path_new)
            print('Success, please manually check both files!')
            return 1


def change_audiosamplerate(path, sr):
    """Change the sample rate of a audio"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_path = os.path.join(temp_dir, os.path.basename(path))
        sys_run_cmd('ffmpeg -y -loglevel fatal -i "%s" -ac 2 -ar %d "%s"' %
                            (path, sr, temp_audio_path))
        if get_samplerate_numaudiosamples_from_audio(temp_audio_path)[0] != sr:
            print("Failed to change audio sample rate!")
            return 2

        shutil.move(temp_audio_path, path)
        print('Success!')
        return 0


def get_samplerate_numaudiosamples(path, sr=None):
    """Get the sample rate of the audio extracted from a video"""
    with tempfile.TemporaryDirectory() as temp_dir:
        sound_file = os.path.join(temp_dir, 'sound.wav')
        sys_run_cmd(\
            'ffmpeg -loglevel fatal -i "{}" -ac 2 {}'.format(path, sound_file))
        y, rate = librosa.load(sound_file, sr=sr)
    return rate, len(y)


def get_samplerate_numaudiosamples_from_audio(path, sr=None):
    y, rate = librosa.load(path, sr=sr)
    return rate, len(y)


def get_duration(path):
    """Get the duration of a video"""
    output = sys_run_cmd_with_stdout(
        'ffprobe -i "{}" -show_entries format=duration -v quiet -of csv="p=0"'.format(path))
    return None if (output is None) else float(output)


def get_duration2(path):
    """Get the duration of a video; more accurate?"""
    output = sys_run_cmd_with_stdout(
        'ffprobe -select_streams v -show_streams "{}" 2>&1'.format(path))
    lines = list(grep_str(output, 'duration='))
    assert len(lines)
    m = re.match(r"duration=(([+-]?(\d+(\.\d*)?)|(\.\d+)))", lines[0])
    return None if (m is None) else float(m.group(1))


def get_duration2_audio(path):
    """Get the duration of a video; more accurate?"""
    output = sys_run_cmd_with_stdout(
        'ffprobe -select_streams a -show_streams "{}" 2>&1'.format(path))
    lines = list(grep_str(output, 'duration='))
    assert len(lines)
    m = re.match(r"duration=(([+-]?(\d+(\.\d*)?)|(\.\d+)))", lines[0])
    return None if (m is None) else float(m.group(1))


def get_numframes(path):
    """Get the number of frames of a video"""
    output = sys_run_cmd_with_stdout(
        'ffprobe -select_streams v -show_streams "{}" 2>&1'.format(path))
    lines = list(grep_str(output, 'nb_frames'))
    assert len(lines)
    m = re.match(r"nb_frames=(\d*)", lines[0])
    return None if (m is None) else int(m.group(1))


def get_timestamps(path, length):
    """Get all the starting timestaps of a video"""
    duration = get_duration(path)
    result = []
    if duration > length:
        result = list(np.arange(0, duration, length/2.))
        while duration - result[-1] < length:
            del result[-1]
    return result


def load_image(path):
    """Load an image"""
    return np.array(Image.open(path))


def plot_spectrogram(snd, sr, plot_path=None, suppress_stdout=False):
    """Draw mel-spectrogram of the given audio"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    S = librosa.feature.melspectrogram(snd, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    # plt.show()

    if plot_path is not None:
        plt.savefig(plot_path)

    plt.close('all')

    if not suppress_stdout:
        print('Spectrogram plot complete.')


def plot_wav(snd, sr, plot_path=None, downsample=False, overlay_mode=False, suppress_stdout=False):
    """Draw waveform plot"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    if downsample:
        librosa.display.waveplot(snd, sr=sr, max_sr=100)
    else:
        librosa.display.waveplot(snd)
    # plt.show()

    if overlay_mode:
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False
        )

    if plot_path is not None:
        plt.savefig(plot_path)

    plt.close('all')

    if not suppress_stdout:
        print('Waveform plot complete.')


def plot_bitstream(bit_stream, plot_path=None, overlay_mode=False, suppress_stdout=False):
    """Draw bitstream"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    y = [int(s) for s in bit_stream]
    # print y
    x = [i for i in range(len(y)+1)]
    # print x

    fig = plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    fig.subplots_adjust(wspace=None, hspace=None)
    
    # plt.plot(x, y, label='bit stream', color="#FF7043")
    # Highlight silent intervals
    from itertools import groupby
    start_idx = 0
    for item in ((k, len(list(g))) for k, g in groupby(y)):
        if item[0] == 0:    # highlight silent intervals
            plt.axvspan(start_idx, start_idx+item[1], color='#FF7043', alpha=0.7)
        elif item[0] == 2:   # highlight excluded intervals
            plt.axvspan(start_idx, start_idx+item[1], color='#78909C', alpha=0.7)
        start_idx += item[1]

    if overlay_mode:
        plt.tick_params(
            axis='y',
            which='both',
            left=False,
            right=False,
            labelleft=False,
            labelright=False
        )

    plt.xlabel('Frames')
    plt.xlim(left=min(x), right=max(x))
    # plt.show()

    if plot_path is not None:
        plt.savefig(plot_path)

    plt.close('all')

    if not suppress_stdout:
        print('Bitstream plot complete.')


def plot_bitstream_animation(bit_stream, animation_path, animation_fps, suppress_stdout=False):
    """Draw bitstream animation"""
    mpl.rcParams['agg.path.chunksize'] = PLOT_CHUNKSIZE
    y = [int(s) for s in bit_stream]
    # print y
    x = [i for i in range(len(y))]
    # print x

    fig = plt.figure(figsize=(PLOT_W, PLOT_H), dpi=PLOT_DPI)
    fig.subplots_adjust(wspace=None, hspace=None)

    # plt.plot(x, y, label='bit stream', color="#FF7043")
    # Highlight silent intervals
    from itertools import groupby
    start_idx = 0
    for item in ((k, len(list(g))) for k, g in groupby(y)):
        if item[0] == 0:    # highlight silent intervals
            plt.axvspan(start_idx, start_idx+item[1]-1, color='#FF7043', alpha=0.7)
        elif item[0] == 2: # highlight excluded intervals
            plt.axvspan(start_idx, start_idx+item[1]-1, color='#78909C', alpha=0.7)
        start_idx += item[1]

    plt.xlim(left=min(x), right=max(x))

    # generate animated plot
    # draw moving vertical line
    X_VALS = x
    padding = 0.02
    min_line = min(y) - padding
    max_line = max(y) + padding

    def update_line(num, line):
        i = X_VALS[num]
        line.set_data([i, i], [min_line, max_line])
        return line,

    l, v = plt.plot(X_VALS[0], min_line, X_VALS[-1], max_line, linewidth=2, color='#00C853')
    line_anim = animation.FuncAnimation(fig, update_line, len(X_VALS), fargs=(l, ))

    plt.grid(False)
    plt.legend()
    plt.xlabel('Frames')
    plt.ylabel('Score')

    # save animated plot
    writer = FFMpegWriter(fps=animation_fps)
    print('Saving animated plot to', animation_path)
    line_anim.save(animation_path, writer=writer)

    # plt.show()

    plt.close('all')

    if not suppress_stdout:
        print('Bitstream animation complete.')


def plot_wav_bitstream_overlay(snd, sr, bit_stream, plot_path=None, suppress_stdout=False):
    """Draw waveform and bitstream together"""
    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = os.path.join(temp_dir, 'wav.png')
        plot_wav(snd, sr, plot_path=wav_path, overlay_mode=True, suppress_stdout=True)
        bit_path = os.path.join(temp_dir, 'bit.png')
        plot_bitstream(bit_stream, bit_path, overlay_mode=True, suppress_stdout=True)

        overlay = Image.open(wav_path)
        background = Image.open(bit_path)

        overlay = overlay.convert("RGBA")
        background = background.convert("RGBA")

        new_img = Image.blend(background, overlay, 0.65)
        # plt.imshow(new_img)

        if plot_path is not None:
            new_img.save(plot_path, "PNG")

        overlay.close()
        background.close()
        new_img.close()

        if not suppress_stdout:
            print('Overlay plot complete.')

'''
def get_bitstream(file_info_dict, begin_padding=VIDEO_BEGIN_PADDING, end_padding=VIDEO_END_PADDING, overwrite=False, suppress_stdout=True):
    """Get the bitstream of a video,  representing its silent/non-silent intervals; 0: silent; 1: non-silent"""
    if not suppress_stdout:
        pprint.pprint(file_info_dict)

    image_dir = os.path.abspath(file_info_dict['path'].split(VIDEO_EXT)[0])
    # print('image_dir:', image_dir)
    return_frames = not os.path.exists(image_dir)
    audio_path = image_dir + AUDIO_EXT
    # print('audio_path:', audio_path)
    return_audio = not os.path.exists(audio_path)
    with VideoFramesAudio(return_frames=return_frames, return_audio=return_audio,\
        output_frame_dir=image_dir, output_audio_path=audio_path, **file_info_dict) as (_, snd_file):
        # load frames
        # ims = np.array(list(map(load_image, img_files)))
        # print(ims.shape, file_info_dict['num_frames'])

        # load audio
        fps = file_info_dict['framerate']
        # print('fps:', fps)
        frames = file_info_dict['num_frames']
        # print('num frames:', frames)
        snd, sr = librosa.load(snd_file, sr=None)
        # print('sr:', sr)

        # Label each frame
        audio_samples_per_frame = int(sr / fps)
        if not suppress_stdout:
            print('Each frame corresponds to {} audio samples'.format(audio_samples_per_frame))
        begin_padding = int(begin_padding*fps*audio_samples_per_frame)
        end_padding = int(len(snd)-end_padding*fps*audio_samples_per_frame)

        # Normalize signal against average
        # print('normalize')
        to_normalize = snd[begin_padding:end_padding]
        if to_normalize.size > 0:
            scale = TARGET_MEAN_SIGNAL_AMPLITUDE / np.mean(np.abs(to_normalize))
            # print('scale:', scale)
            snd_normalized = np.clip(to_normalize * scale, -1., 1.)
            snd[begin_padding:end_padding] = snd_normalized
            # print('normalize done')
        else:
            begin_padding = 0
            end_padding = len(snd)
            scale = TARGET_MEAN_SIGNAL_AMPLITUDE / np.mean(np.abs(snd))
            # print('scale:', scale)
            snd = np.clip(snd * scale, -1., 1.)
            # print('normalize done')
        # print('file:', file_info_dict['path'])
        # print('avg signal:', np.mean(np.abs(snd[begin_padding:end_padding])))
        # print('avg signal:', np.mean(np.abs(snd)))

        # Threshold of wave signal where below it would be marked as 0 and higher it would be marked as 1
        wav_threshold = 0.008
        mask = np.where(np.abs(snd) < wav_threshold, 0, 1)  # compare each time frame to threshold

        cnt = 0
        output = ''
        # Threshold of ratio of 0/1 within a chunk of one frame worth of time long
        # If ratio > threshold, then the current interval is silent and vice versa
        ratio_threshold = 0.25

        for i in range(0, len(mask), audio_samples_per_frame):
            cnt += 1

            if i < begin_padding or i >= end_padding:
                output += '2'
            else:
                start = i
                end = i + audio_samples_per_frame
                end = len(mask) if end > len(mask) else end
                # print('{}: ({}, {}) => Diff: {}'.format(cnt, start, end, end-start))
                chunk = mask[start:end]
                ratio = float((len(chunk)-sum(chunk))/len(chunk))
                # print('Frame {}: {} / {} = {} {} {} ==> {}'.format(\
                #     i/audio_samples_per_frame, len(chunk)-sum(chunk), len(chunk), ratio,\
                #         ('<' if ratio < ratio_threshold else ('=' if ratio == ratio_threshold else '>')),\
                #             ratio_threshold, ('silent' if ratio > ratio_threshold else 'non-silent')))
                output += '0' if ratio > ratio_threshold else '1'

            if cnt >= frames:
                break

        # Draw spectrogram
        # spectro_path = create_path_from_filepath(file_info_dict['path'], 'spectrogram.png')
        # plot_spectrogram(snd, sr, plot_path=spectro_path)
        # Draw waveform
        # wav_path = create_path_from_filepath(file_info_dict['path'], 'waveform.png')
        # plot_wav(snd, sr, plot_path=wav_path)
        # Draw bitstream animation
        # bitani_path = create_path_from_filepath(file_info_dict['path'], 'animated.mp4')
        # plot_bitstream_animation(output, bitani_path, fps)
        # Draw bitstream
        # bit_path = create_path_from_filepath(file_info_dict['path'], 'bitstream.png')
        # plot_bitstream(output, plot_path=bit_path, suppress_stdout=True)
        # Draw waveform and bitstream overlay
        wav_bit_path = create_path_from_filepath(file_info_dict['path'], 'overlay.png')
        if (not os.path.exists(wav_bit_path)) or overwrite:
            plot_wav_bitstream_overlay(snd, sr, output, wav_bit_path, True)

        # Calculate silence_total_ratio
        # len(silent intervals) / len(video)
        s_t_r = output.count('0') / len(output)

        # Calculate avg_silenceInterval_silcenceTotal_ratio
        # avg(each len(silent interval) / len(silent intervals))
        a_s_s_r = 0 if output.count('0') == 0\
            else np.mean([len(list(g)) for k, g in groupby(output) if k == '0'])\
            /output.count('0')

    return output, s_t_r, a_s_s_r, image_dir, audio_path


def get_bitstream_better(file_info_dict, begin_padding=VIDEO_BEGIN_PADDING, end_padding=VIDEO_END_PADDING,\
    overwrite=False, postfix='', filtered=False, extract_face=False, extract_flow=False, suppress_stdout=True):
    """
        Get the bitstream of a video,  representing its silent/non-silent intervals; 0: silent; 1: non-silent.
        Better implementation: one threshold; bit label depends on surrounding bits
    """
    if not suppress_stdout:
        pprint.pprint(file_info_dict)

    image_dir = os.path.abspath(file_info_dict['path'].split(VIDEO_EXT)[0])
    # print('image_dir:', image_dir)
    return_frames = (not os.path.exists(image_dir)) or overwrite
    # print('return_frames:', return_frames)
    audio_path = image_dir + AUDIO_EXT
    # print('audio_path:', audio_path)
    return_audio = (not os.path.exists(audio_path)) or overwrite
    # return_audio = True # should always be True so that bitstream can be calculated
    # print('return_audio:', return_audio)
    with VideoFramesAudio(return_frames=return_frames, return_audio=return_audio,\
        output_frame_dir=image_dir, output_audio_path=audio_path,\
            extract_face=extract_face, extract_flow=extract_flow, overwrite=overwrite,\
                **file_info_dict) as (faces_dir, flows_dir, snd_file):
        # load frames
        # ims = np.array(list(map(load_image, img_files)))
        # print(ims.shape, file_info_dict['num_frames'])

        # load audio
        fps = file_info_dict['framerate']
        # print('fps:', fps)
        frames = file_info_dict['num_frames']
        # print('num frames:', frames)
        # print('snd_file:', snd_file)
        snd, sr = librosa.load(snd_file, sr=file_info_dict['audio_sample_rate'])
        # print('sr:', sr)

        # Label each frame
        audio_samples_per_frame = int(sr / fps)
        if not suppress_stdout:
            print('Each frame corresponds to {} audio samples'.format(audio_samples_per_frame))
        begin_padding = int(begin_padding*fps*audio_samples_per_frame)
        end_padding = int(len(snd)-end_padding*fps*audio_samples_per_frame)

        # Normalize signal against average
        # print('normalize')
        to_normalize = snd[begin_padding:end_padding]
        if to_normalize.size > 0:
            scale = TARGET_MEAN_SIGNAL_AMPLITUDE / np.mean(np.abs(to_normalize))
            # print('scale:', scale)
            snd_normalized = np.clip(to_normalize * scale, -1., 1.)
            snd[begin_padding:end_padding] = snd_normalized
            # print('normalize done')
        else:
            begin_padding = 0
            end_padding = len(snd)
            scale = TARGET_MEAN_SIGNAL_AMPLITUDE / np.mean(np.abs(snd))
            # print('scale:', scale)
            snd = np.clip(snd * scale, -1., 1.)
            # print('normalize done')
        # print('file:', file_info_dict['path'])
        # print('avg signal:', np.mean(np.abs(snd[begin_padding:end_padding])))
        # print('avg signal:', np.mean(np.abs(snd)))

        # Threshold of wave signal where below it would be marked as 0 and higher it would be marked as 1
        # wav_threshold = 0.008
        # mask = np.where(np.abs(snd) < wav_threshold, 0, 1)  # compare each time frame to threshold

        cnt = 0
        # output = ''
        # Threshold of ratio of 0/1 within a chunk of one frame worth of time long
        # If ratio > threshold, then the current interval is silent and vice versa
        # ratio_threshold = 0.25

        snd_avg_per_video_frame = []

        for i in range(0, len(snd), audio_samples_per_frame):
            cnt += 1

            if i < begin_padding or i >= end_padding:
                # output += '2'
                snd_avg_per_video_frame.append(2)
            else:
                start = i
                end = i + audio_samples_per_frame
                end = len(snd) if end > len(snd) else end
                # print('{}: ({}, {}) => Diff: {}'.format(cnt, start, end, end-start))
                chunk = snd[start:end]
                # ratio = float((len(chunk)-sum(chunk))/len(chunk))
                # # print('Frame {}: {} / {} = {} {} {} ==> {}'.format(\
                # #     i/audio_samples_per_frame, len(chunk)-sum(chunk), len(chunk), ratio,\
                # #         ('<' if ratio < ratio_threshold else ('=' if ratio == ratio_threshold else '>')),\
                # #             ratio_threshold, ('silent' if ratio > ratio_threshold else 'non-silent')))
                # output += '0' if ratio > ratio_threshold else '1'

                # snd_avg_per_video_frame.append(np.mean(np.abs(chunk)))    # mean
                snd_avg_per_video_frame.append(np.linalg.norm(chunk))    # 2-norm
            if cnt >= frames:
                break

        output = ['2']*len(snd_avg_per_video_frame)
        if filtered:
            output_filtered = output.copy()

        indices = truncate(snd_avg_per_video_frame)
        to_process = np.array(snd_avg_per_video_frame[indices[0]:indices[1]])
        # print(to_process)

        # normalize 2-norms
        to_process /= np.max(to_process)
        assert np.min(to_process) >= 0 and np.max(to_process) <= 1

        # NOT FILTER
        score_threshold = 0.08
        for i in range(len(to_process)):
            score = to_process[i]
            # print(score)
            output[i+indices[0]] = '0' if score < score_threshold else '1'
        output = ''.join(output)
        # print(output)

        # FILTER
        if filtered:
            # sigma = 1
            # to_process = list(ndimage.filters.gaussian_filter1d(to_process, sigma))

            window_length = 7
            to_process_padded = np.pad(to_process, window_length//2, 'reflect')
            # print(to_process_padded)

            score_threshold_filtered = 0.08
            for i in range(len(to_process_padded)):
                start = i
                end = i + window_length
                if end > len(to_process_padded):
                    break
                chunk = to_process_padded[start:end]
                # print(chunk)
                score = dot(gauss_weights(len(chunk)), chunk)
                # print(score)
                output_filtered[i+indices[0]] = '0' if score < score_threshold_filtered else '1'
            output_filtered = ''.join(output_filtered)
            # print(output_filtered)

        # Draw spectrogram
        # spectro_path = create_path_from_filepath(file_info_dict['path'], 'spectrogram.png')
        # plot_spectrogram(snd, sr, plot_path=spectro_path)
        # Draw waveform
        # wav_path = create_path_from_filepath(file_info_dict['path'], 'waveform.png')
        # plot_wav(snd, sr, plot_path=wav_path)
        # Draw bitstream animation
        # bitani_path = create_path_from_filepath(file_info_dict['path'], 'animated.mp4')
        # plot_bitstream_animation(output, bitani_path, fps)
        # Draw bitstream
        # bit_path = create_path_from_filepath(file_info_dict['path'], 'bitstream.png')
        # plot_bitstream(output, plot_path=bit_path, suppress_stdout=True)
        # Draw waveform and bitstream overlay
        wav_bit_path = create_path_from_filepath(file_info_dict['path'], 'overlay' + postfix + '.png')
        if (not os.path.exists(wav_bit_path)) or overwrite:
            plot_wav_bitstream_overlay(snd, sr, output, wav_bit_path, True)

        if filtered:
            wav_bit_path = create_path_from_filepath(file_info_dict['path'], 'overlay' + postfix + '_filtered.png')
            if (not os.path.exists(wav_bit_path)) or overwrite:
                plot_wav_bitstream_overlay(snd, sr, output_filtered, wav_bit_path, True)

        # Calculate silence_total_ratio
        # len(silent intervals) / len(video)
        s_t_r = output.count('0') / len(output)

        # Calculate avg_silenceInterval_silcenceTotal_ratio
        # avg(each len(silent interval) / len(silent intervals))
        a_s_s_r = 0 if output.count('0') == 0\
            else np.mean([len(list(g)) for k, g in groupby(output) if k == '0'])\
            /output.count('0')

    if filtered:
        return output, output_filtered, s_t_r, a_s_s_r, image_dir, faces_dir, flows_dir, audio_path
    else:
        print(output, s_t_r, a_s_s_r, image_dir, faces_dir, flows_dir, audio_path)
        return output, s_t_r, a_s_s_r, image_dir, faces_dir, flows_dir, audio_path


def video_has_audio(path):
    """Check whether a video has audio"""
    output = sys_run_cmd_with_stdout('ffmpeg -i "{}" 2>&1'.format(path))
    matches = list(grep_str(output, 'Stream #0:1'))
    if len(matches) > 0:
        return 'Audio: none' not in matches[0]
    else:
        return False


class VideoFramesAudio:
    """Class separates video into frames and audio"""
    def __init__(self, path, framerate=None, num_frames=None,\
                audio_sample_rate=None, duration=None,\
                    start_time=None, end_time=None,\
                        return_frames=False, return_audio=False,\
                            output_frame_dir=None, output_audio_path=None,\
                                extract_face=False, extract_flow=False, face_x=None, face_y=None,\
                                    overwrite=False, **kargs):
        self.temp_dir = None
        self.path = path
        self.fps = framerate
        self.num_frames = num_frames
        self.sr = audio_sample_rate
        self.duration = duration
        self.start_time = start_time
        self.end_time = end_time
        self.return_frames = return_frames
        self.return_audio = return_audio
        # print(self.return_frames)
        # print(self.return_audio)
        self.output_frame_dir = output_frame_dir
        self.output_audio_path = output_audio_path
        self.extract_face = extract_face
        self.extract_flow = extract_flow and extract_face
        self.face_x = face_x
        self.face_y = face_y
        self.overwrite = overwrite

        self.start_str = ''
        if self.start_time is not None and self.end_time is not None:
            self.dur_str = '-t %f' % (self.end_time - max(0, self.start_time)) if self.end_time is not None else ''
            self.start_str = '-ss %f' % self.start_time if self.start_time is not None else ''
        else:
            self.dur_str = '-t %f' % self.duration if self.duration is not None else ''
        self.fps_str = '-r %f' % self.fps if self.fps is not None else ''

    def __enter__(self):
        if not os.path.exists(self.path):
            raise RuntimeError('Video does not exist:', self.path)

        self.temp_dir = tempfile.mkdtemp()
        temp_img_dir = os.path.join(self.temp_dir, 'image')
        if self.output_frame_dir is not None:
            temp_img_dir = self.output_frame_dir
        temp_snd_dir = os.path.join(self.temp_dir, 'sound')
        sound_file = os.path.join(temp_snd_dir, 'sound.wav')
        if self.output_audio_path is not None:
            sound_file = self.output_audio_path

        if self.return_frames:
            # temp_img_dir = os.path.join(self.temp_dir, 'image')
            create_folder(temp_img_dir)
            if self.start_str == '':
                sys_run_cmd('ffmpeg -loglevel fatal -i "%s" %s %s "%s/%%07d.jpg"' %
                            (self.path, self.dur_str, self.fps_str, temp_img_dir))
            else:
                sys_run_cmd('ffmpeg -y -loglevel fatal %s -i "%s" %s %s "%s/%%07d.jpg"' %
                            (self.start_str, self.path, self.dur_str, self.fps_str, temp_img_dir))
            frame_files = sorted(glob.glob((os.path.join(temp_img_dir, '*.jpg'))))

            if self.extract_face:
                initial_bb = get_initial_bb(frame_files[0], self.face_x, self.face_y)
                # success, faces_dir = extract_faces_by_bb(*get_time_series(temp_img_dir, initial_bb=initial_bb), out_face_size=256, overwrite=overwrite)
                success, faces_dir = extract_all_faces(temp_img_dir, initial_bb=initial_bb, out_face_size=256, overwrite=self.overwrite)

            if self.extract_flow:
                if self.extract_face:
                    flows_dir, _ = extract_flow(faces_dir)
                    convert_flow(flows_dir)
        else:
            if os.path.exists(temp_img_dir):
                frame_files = sorted(glob.glob((os.path.join(temp_img_dir, '*.jpg'))))

                if self.extract_face:
                    initial_bb = get_initial_bb(frame_files[0], self.face_x, self.face_y)
                    # success, faces_dir = extract_faces_by_bb(*get_time_series(temp_img_dir, initial_bb=initial_bb), out_face_size=256, overwrite=overwrite)
                    success, faces_dir = extract_all_faces(temp_img_dir, initial_bb=initial_bb, out_face_size=256, overwrite=self.overwrite)

                if self.extract_flow:
                    if self.extract_face:
                        flows_dir, _ = extract_flow(faces_dir)
                        convert_flow(flows_dir)
            else:
                temp_img_dir = None
                frame_files = None

        if self.return_audio:
            if video_has_audio(self.path):
                # temp_snd_dir = os.path.join(self.temp_dir, 'sound')
                create_folder(temp_snd_dir)
                # sound_file = os.path.join(temp_snd_dir, 'sound.wav')
                sys_run_cmd('ffmpeg -y -loglevel fatal -i "%s" %s -ac 2 -ar %d "%s"' %
                            (self.path, self.dur_str, self.sr, sound_file))
            else:
                sound_file = None
        else:
            sound_file = self.output_audio_path

        if not self.extract_face and not self.extract_flow:
            return temp_img_dir, None, sound_file
        elif self.extract_face and not self.extract_flow:
            return faces_dir, None, sound_file
        elif self.extract_face and self.extract_flow:
            return faces_dir, flows_dir, sound_file

    def __exit__(self, type, value, tb):
        shutil.rmtree(self.temp_dir)
'''