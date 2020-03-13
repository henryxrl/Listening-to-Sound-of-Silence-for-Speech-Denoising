import gc
import json
import math
import os
import shutil
import subprocess
from collections import OrderedDict
from itertools import groupby
import numpy as np


PROJECT_ROOT = "../"
DATA_ROOT = os.path.join(os.path.abspath(PROJECT_ROOT), "data")
JSON_DUMP_PARAMS = dict(indent=4, sort_keys=False, ensure_ascii=False, separators=(',', ':'))
VIDEO_EXT = '.mp4'
AUDIO_EXT = '.wav'
VIDEO_BEGIN_PADDING = 15    # seconds
VIDEO_END_PADDING = 15    # seconds
TARGET_MEAN_SIGNAL_AMPLITUDE = 0.05
PLOT_W = 50
PLOT_H = 10
PLOT_DPI = 100
PLOT_CHUNKSIZE = 1e10

FRAMERATE = 30
AUDIO_SAMPLE_RATE = 44100

FIELDS = ['path', 'framerate', 'audio_sample_rate', 'audio_samples', 'duration',\
    'num_frames', 'bit_stream', 'silence_total_ratio',\
        'avg_silenceInterval_silcenceTotal_ratio', 'frames_path',\
            'face_x', 'face_y', 'clip_start_time', 'clip_end_time', 'audio_path', 'flows_path']


def ordered_dict_insert(dct, key, value, key_insert_after, dict_setitem=dict.__setitem__):
    """Insert a key value pair into a dictionary at a given position (key after which to be inserted)"""
    dct.update({key:value})
    met_key = False
    for k in list(dct.keys()):
        if k == key_insert_after:
            met_key = True
        if met_key and k != key_insert_after and k != key:
            dct.move_to_end(k, last=True)


def gauss_weights(n=5, sigma=1):
    """Generate 1D Gaussian kernel as weights"""
    if n % 2 == 0:
        r = np.linspace(-int(n/2)+0.5, int(n/2)-0.5, n)
    else:
        r = np.linspace(-int(n/2), int(n/2), n)
    weights = [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]
    return [x / sum(weights) for x in weights]


def dot(K, L):
    """Dot product of two lists"""
    if len(K) != len(L):
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))


def truncate(l):
    """Truncate the leading and trailing '2' floats"""
    indices = [len(list(g)) for k, g in groupby(l) if k == 2]
    if not indices:
        return (0, len(l))
    return (indices[0], -indices[1])


def magnitude(x):
    """Calculate the order of magnitude of any number"""
    return int(math.log10(x))


def isclose(f1, f2):
    """Check whether two numbers are close"""
    f1_m = magnitude(f1)
    f2_m = magnitude(f2)
    if f1_m == f2_m:
        f1_sig = f1 / math.pow(10, f1_m)
        f2_sig = f2 / math.pow(10, f2_m)
        return np.isclose(f1_sig, f2_sig, atol=1.e-1)
    return False


def print_dictionary(dictionary, sep=',', key_list=None, omit_keys=False):
    if key_list is None:
        key_list = dictionary.keys()
    print('{', end='')
    for idx, key in enumerate(key_list):
        end_str = sep + ' ' if idx < len(key_list)-1 else ''
        if not omit_keys:
            print('\'{}\': \'{}\''.format(key, dictionary[key]), end=end_str)
        else:
            print('\'{}\''.format(dictionary[key]), end=end_str)
    print('}')


def get_number_of_files(dictionary, ext=None):
    """Count number of files in a dictionary"""
    if ext is None:
        return len([name for name in os.listdir(dictionary)\
            if os.path.isfile(os.path.join(dictionary, name))])
    else:
        return len([name for name in os.listdir(dictionary)\
            if os.path.isfile(os.path.join(dictionary, name)) and name.endswith(ext)])


def ensure_dir(directory):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_folder(directory):
    """Create folder; remove first if already exists"""
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)


def create_path_from_filepath(input_filepath, trailing_name):
    """Create new file path from input file path"""
    return os.path.join(os.path.abspath(\
            os.path.join(input_filepath, os.pardir)),\
                os.path.basename(input_filepath).split(VIDEO_EXT)[0] + '_' + trailing_name)


def get_parent_dir(path):
    """Get parent directory"""
    return os.path.abspath(os.path.join(path, os.pardir))


def get_path_same_dir(path, new_file_name):
    """Get the absolute path of a new file that is in the same directory as another file"""
    return os.path.join(get_parent_dir(path), new_file_name)


def sys_run_cmd(cmd):
    """Execute shell command"""
    gc.collect()
    return subprocess.Popen(cmd, shell=True).wait()


def sys_run_cmd_with_stdout(cmd):
    """Execute shell command and return string containing stdout result"""
    gc.collect()
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()


def grep_str(text, query):
    """Query substring from string"""
    # does not handle regexp
    for line in text.decode().split('\n'):
        if query in line:
            yield line


def get_num_files(path, ext=VIDEO_EXT):
    """Get the number of files inside a directory"""
    # return sum([len([f for f in files if f.endswith(ext)\
    #      and video_has_audio(os.path.join(path, f))])\
    #          for r, d, files in os.walk(path)])
    return sum([len([f for f in files if f.endswith(ext)]) for r, d, files in os.walk(path)])


def combine_alljson(json_dir, output_json_path, exclude_yids=None):
    """Combine all individual json files into a single json file with exclusion"""
    if exclude_yids is not None:
        if not isinstance(exclude_yids, list):
            print('Invalid exclusive Youtube id list')
            return

    all_json_info = []
    for file in os.listdir(json_dir):
        if file.endswith('.json'):
            filepath = os.path.join(json_dir, file)
            file_yid = os.path.basename(file).split('.json')[0]
            if exclude_yids is None or file_yid not in exclude_yids:
                with open(filepath, 'r') as fp:
                    all_json_info.append(json.load(fp))

    json_clips = []
    for info in all_json_info:
        # print(info['num_videos'])
        json_clips += info['files']
    json_clips = sorted(json_clips, key=lambda x:os.path.basename(x['path']))
    print('Num videos: {}'.format(len(json_clips)))

    # build dictionary
    json_dict = OrderedDict()
    json_dict['dataset_path'] = os.path.normpath(os.path.join(all_json_info[0]['dataset_path'], os.pardir))
    json_dict['num_videos'] = len(json_clips)
    json_dict['files'] = json_clips

    # write file
    print('Writing to "{}"...'.format(output_json_path))
    with open(output_json_path, 'wb') as f:
        json_str = json.dumps(json_dict, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())

    print('Done')
