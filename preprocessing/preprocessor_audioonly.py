import json
import os
from tqdm import tqdm
import shutil
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from util import *


def process_audio(csv_path, output_dir, audio_id, output_json, ext=AUDIO_EXT, print_progress=True):
    """Process an audio file, or all the audio files with the same audio_id
    
    Arguments:
        csv_path {str} -- Path to the csv file describing the dataset
        output_dir {str} -- Path to the output directory containing the processed audio file(s)
        audio_id {str} -- Audio id of the to-be-processed audio file
        output_json {str} -- Path to the json file containing detailed information about every audio file in the output directory
    
    Keyword Arguments:
        ext {str} -- Audio extension (default: {AUDIO_EXT})
        print_progress {bool} -- Print progress when True (default: {True})
    
    Returns:
        int -- Return the number of failed audio files
    """

    print('Processing "{}"...'.format(audio_id))
    ensure_dir(output_dir)
    df = pd.read_csv(csv_path, header=None)
    # print(df)
    df_todownload = df.loc[df.iloc[:, 0] == audio_id.split('/')[0]]
    # print(df_todownload)
    num_videos = df_todownload.shape[0]
    if num_videos == 0:
        return -1
    hierarchy = OrderedDict([
        ('dataset_path', output_dir),
        ('num_videos', num_videos)
    ])
    file_info_list = []

    # download the whole video first
    whole_video_path = os.path.join(get_parent_dir(csv_path), audio_id + ext)
    # whole_video_success = youtube_dl_full(audio_id, whole_video_path, FRAMERATE)
    whole_video_success = os.path.exists(whole_video_path)
    # print(whole_video_path, whole_video_success)
    if whole_video_success:
        count = 0
        success_count = 0
        field_toprint = [x for x in FIELDS if x not in (
            'bit_stream', 'frames_path')]
        for index, row in df_todownload.iterrows():
            file_info = OrderedDict()
            if print_progress:
                print('To process: ', row)
            path = os.path.join(
                output_dir, '{}_{:07}{}'.format(row[0], count+1, ext))
            duration = get_duration2_audio(whole_video_path)
            shutil.move(whole_video_path, path)
            file_info[FIELDS[0]] = path
            file_info[FIELDS[12]] = 0
            file_info[FIELDS[13]] = duration
            file_info[FIELDS[10]] = 0
            file_info[FIELDS[11]] = 0
            file_info[FIELDS[1]] = FRAMERATE
            (file_info[FIELDS[2]], file_info[FIELDS[3]]) = \
                get_samplerate_numaudiosamples(path, sr=AUDIO_SAMPLE_RATE)
            change_audiosamplerate(path, AUDIO_SAMPLE_RATE)
            file_info[FIELDS[4]] = duration
            file_info[FIELDS[5]] = math.ceil(duration*FRAMERATE)
            file_info[FIELDS[6]] = '1' * file_info[FIELDS[5]]
            file_info[FIELDS[7]] = 0
            file_info[FIELDS[8]] = 0
            file_info[FIELDS[9]] = None
            file_info[FIELDS[15]] = None
            file_info[FIELDS[14]] = path

            file_info_list.append(file_info)
            count += 1
            success_count += 1
            # print('11')
            if print_progress:
                print_dictionary(file_info, key_list=field_toprint)
                print(
                    '========== {} / {} Complete ==========\n'.format(count, num_videos))

        hierarchy['files'] = file_info_list
        json_str = json.dumps(hierarchy, **JSON_DUMP_PARAMS)
        # print(json_str)

        # write file
        if print_progress:
            print('Writing to "{}"...'.format(output_json))
        with open(output_json, 'wb') as f:
            f.write(json_str.encode())
        if print_progress:
            print('Done\n')

        # remove whole video
        if os.path.exists(whole_video_path):
            os.remove(whole_video_path)

        return num_videos-success_count

    else:
        return num_videos


def build_json_better(download_dataset_dir, download_dataset_csv, output_json, filenamewithouthext=None, ext=AUDIO_EXT, print_progress=True):
    """Build dataset json which contains detailed information about every audio file in the dataset
    
    Arguments:
        download_dataset_dir {str} -- Path to the dataset directory containing all the to-be-processed audio files
        download_dataset_csv {str} -- Path to the csv file describing the dataset
        output_json {str} -- Path to the final generated dataset json file
    
    Keyword Arguments:
        filenamewithouthext {str} -- If not None, only look for the files containing the specified file name (default: {None})
        ext {str} -- Audio extension (default: {AUDIO_EXT})
        print_progress {bool} -- Print progress when True (default: {True})
    
    Returns:
        int -- Return the number of failed audio files
    """

    total_failed = 0

    if filenamewithouthext is None:
        video_yids = [f.stem for f in Path(
            download_dataset_dir).rglob('*' + ext)]
    else:
        video_yids = [os.path.join(get_parent_dir(f).split('/')[-1], f.stem) for f in Path(
            download_dataset_dir).rglob('*' + ext) if filenamewithouthext in f.name]

    if filenamewithouthext is None:
        res = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(process_audio)
                                                             (download_dataset_csv, os.path.join(download_dataset_dir, yid),
                                                              yid, os.path.join(download_dataset_dir, '{}.json'.format(yid)),
                                                              ext, print_progress) for yid in video_yids)
        total_failed += sum(res)
    else:
        res = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(process_audio)
                                                             (download_dataset_csv, os.path.join(download_dataset_dir, yid.split('/')[0]),
                                                              yid, os.path.join(download_dataset_dir, '{}.json'.format(yid.split('/')[0])),
                                                              ext, print_progress) for yid in video_yids)
        total_failed += sum(res)

    if total_failed > 0:
        print('Failure to download {} video clips'.format(total_failed))
        print('Aborted')
    else:
        print('Download complete')
        print('Writing dataset json: {}...'.format(output_json))
        combine_alljson(download_dataset_dir, output_json)

    return total_failed


def build_csv(download_dataset_dir, output_csv, ext=AUDIO_EXT):
    with open(output_csv, 'w') as f:
        for path in Path(download_dataset_dir).rglob('*'+ext):
            f.write(str(path.stem) + '\n')
    print('DONE')


if __name__ == "__main__":
    # SOS_DIR = os.path.join(DATA_ROOT, 'sounds_of_silence_audioonly')
    # SOS_CSV = os.path.join(SOS_DIR, 'sounds_of_silence.csv')
    # SOS_JSON = os.path.join(DATA_ROOT, 'sounds_of_silence.json')
    # build_json_better(SOS_DIR, SOS_CSV, SOS_JSON)

    # DIR = '/proj/vondrick/rx2132/test_noise_robust_embedding/data/TIMIT/TEST_noisy'
    # CSV = '/proj/vondrick/rx2132/test_noise_robust_embedding/data/TIMIT/TEST_noisy/TEST_noisy.csv'
    # # build_csv(DIR, CSV, ext='.WAV')
    # JSON = '/proj/vondrick/rx2132/test_noise_robust_embedding/data/TIMIT/TEST_noisy.json'
    # build_json_better(DIR, CSV, JSON, ext='.WAV')

    SNR = [-10, -7, -3, 0, 3, 7, 10]
    for snr in tqdm(SNR):
        DIR = '/proj/vondrick/rx2132/test_noise_robust_embedding/data/TIMIT/TEST_noisy_snr' + str(int(snr))
        CSV = '/proj/vondrick/rx2132/test_noise_robust_embedding/data/TIMIT/TEST_noisy_snr' + str(int(snr)) + '/TEST_noisy_snr' + str(int(snr)) + '.csv'
        build_csv(DIR, CSV, ext='.WAV')
        JSON = '/proj/vondrick/rx2132/test_noise_robust_embedding/data/TIMIT/TEST_noisy_snr' + str(int(snr)) + '.json'
        build_json_better(DIR, CSV, JSON, ext='.WAV')
