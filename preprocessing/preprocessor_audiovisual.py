import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from util import *


def process_video(csv_path, output_dir, video_id, output_json, print_progress=True):
    """Process an video file, or all the video files with the same video_id
    
    Arguments:
        csv_path {str} -- Path to the csv file describing the dataset
        output_dir {str} -- Path to the output directory containing the processed video file(s)
        video_id {str} -- Video id of the to-be-processed video file
        output_json {str} -- Path to the json file containing detailed information about every video file in the output directory
    
    Keyword Arguments:
        print_progress {bool} -- Print progress when True (default: {True})
    
    Returns:
        int -- Return the number of failed audio files
    """

    print('Processing "{}"...'.format(video_id))
    ensure_dir(output_dir)

    df = pd.read_csv(csv_path, header=None)
    df_todownload = df.loc[df.iloc[:, 0] == video_id]
    # print(df_todownload.shape)
    num_videos = df_todownload.shape[0]
    if num_videos == 0:
        return -1

    hierarchy = OrderedDict([
        ('dataset_path', output_dir),
        ('num_videos', num_videos)
    ])
    file_info_list = []

    # download the whole video first
    whole_video_path = get_path_same_dir(csv_path, video_id + VIDEO_EXT)
    # whole_video_success = youtube_dl_full(video_id, whole_video_path, FRAMERATE)
    whole_video_success = os.path.exists(whole_video_path)
    # print(whole_video_path, whole_video_success)
    if whole_video_success:
        count = 0
        success_count = 0
        field_toprint = [x for x in FIELDS if x not in ('bit_stream', 'frames_path')]
        for index, row in df_todownload.iterrows():
            file_info = OrderedDict()
            if print_progress:
                print('To download: ', row)
            path = os.path.join(output_dir, '{}_{:07}{}'.format(row[0], count+1, VIDEO_EXT))
            # # success = youtube_dl(row[0], row[1], row[2], path, FRAMERATE)
            # success = cut_video(whole_video_path, row[1], row[2], path, FRAMERATE)
            duration = get_duration2(whole_video_path)
            if float(row[2]) == -1:
                end = duration - float(row[1])
            else:
                end = float(row[2])
            if float(row[1]) == 0 and end == duration:
                shutil.move(whole_video_path, path)
                success = True
            else:
                print('Cut video')
                success = cut_video(whole_video_path, float(row[1]), float(row[2]), path, FRAMERATE)
            if success:
                file_info[FIELDS[0]] = path
                file_info[FIELDS[12]] = float(row[1])
                file_info[FIELDS[13]] = end
                file_info[FIELDS[10]] = float(row[3])
                file_info[FIELDS[11]] = float(row[4])
                change_framerate(path, fr=FRAMERATE)
                file_info[FIELDS[1]] = get_framerate(path)
                (file_info[FIELDS[2]], file_info[FIELDS[3]]) = \
                    get_samplerate_numaudiosamples(path, sr=AUDIO_SAMPLE_RATE)
                file_info[FIELDS[4]] = duration
                file_info[FIELDS[5]] = get_numframes(path)
                (file_info[FIELDS[6]], file_info[FIELDS[7]],\
                    file_info[FIELDS[8]], _, file_info[FIELDS[9]],\
                        file_info[FIELDS[15]], file_info[FIELDS[14]]) =\
                            get_bitstream_better(file_info, begin_padding=0, end_padding=0, extract_face=True, extract_flow=False, overwrite=False)

                file_info_list.append(file_info)
                count += 1
                success_count += 1
                if print_progress:
                    print_dictionary(file_info, key_list=field_toprint)
                    print('========== {} / {} Complete ==========\n'.format(count, num_videos))
            else:
                count += 1
                if print_progress:
                    print('Failed to download {}.{}'.format(row[0], VIDEO_EXT))

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


def build_json_better(download_dataset_dir, download_dataset_csv, output_json, print_progress=True):
    """Build dataset json which contains detailed information about every video file in the dataset
    
    Arguments:
        download_dataset_dir {str} -- Path to the dataset directory containing all the to-be-processed audio files
        download_dataset_csv {str} -- Path to the csv file describing the dataset
        output_json {str} -- Path to the final generated dataset json file
    
    Keyword Arguments:
        print_progress {bool} -- Print progress when True (default: {True})
    
    Returns:
        int -- Return the number of failed audio files
    """

    total_failed = 0

    video_yids = [f.stem for f in Path(download_dataset_dir).rglob('*' + VIDEO_EXT)]

    res = Parallel(n_jobs=-1, backend="multiprocessing")\
        (delayed(process_video)\
            (download_dataset_csv, os.path.join(download_dataset_dir, yid),\
                yid, os.path.join(download_dataset_dir, '{}.json'.format(yid)),\
                    print_progress) for yid in video_yids)
    total_failed += sum(res)

    if total_failed > 0:
        print('Failure to download {} video clips'.format(total_failed))
        print('Aborted')
    else:
        print('Download complete')
        print('Writing dataset json: {}...'.format(output_json))
        combine_alljson(download_dataset_dir, output_json)

    return total_failed


if __name__ == "__main__":
    # HENRIQUE_DIR = os.path.join(DATA_ROOT, 'henrique_audiovisual')
    # HENRIQUE_CSV = os.path.join(HENRIQUE_DIR, 'henrique.csv')
    # HENRIQUE_JSON = os.path.join(DATA_ROOT, 'henrique_audiovisual.json')
    # build_json_better(HENRIQUE_DIR, HENRIQUE_CSV, HENRIQUE_JSON)

    # LANGUAGES_DIR = os.path.join(DATA_ROOT, 'languages_audiovisual')
    # LANGUAGES_CSV = os.path.join(LANGUAGES_DIR, 'languages.csv')
    # LANGUAGES_JSON = os.path.join(DATA_ROOT, 'languages_audiovisual.json')
    # build_json_better(LANGUAGES_DIR, LANGUAGES_CSV, LANGUAGES_JSON)

    # CE_DIR = os.path.join(DATA_ROOT, 'counterexamples_audiovisual')
    # CE_CSV = os.path.join(CE_DIR, 'counterexamples.csv')
    # CE_JSON = os.path.join(DATA_ROOT, 'counterexamples_audiovisual.json')
    # build_json_better(CE_DIR, CE_CSV, CE_JSON)

    RW_DIR = os.path.join(DATA_ROOT, 'real_world_audiovisual')
    RW_CSV = os.path.join(RW_DIR, 'real_world.csv')
    RW_JSON = os.path.join(DATA_ROOT, 'real_world_audiovisual.json')
    build_json_better(RW_DIR, RW_CSV, RW_JSON)
