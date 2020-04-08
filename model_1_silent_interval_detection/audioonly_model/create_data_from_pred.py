import argparse
import errno
import json
import os
import numpy as np
from shutil import copy2

from collections import OrderedDict
from tqdm import tqdm

from operator import itemgetter
from itertools import groupby
from sklearn.preprocessing import minmax_scale

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

from dataset import CLIP_FRAMES, SILENT_CONSECUTIVE_FRAMES, DATA_REQUIRED_SR
from utils import print_dictionary, find_common_path, ensure_dir
from tools import *
from predict import EXPERIMENT_PREDICTION_OUTPUT_DIR, HENRIQUE_JSON, LANGUAGES_JSON, LOOKING_TO_LISTEN_JSON, CE_JSON


def create_data_from_prediction(input_json, output_json=None):
    if output_json is None:
        output_json = os.path.join(os.path.abspath(os.path.join(input_json, os.pardir)), 'pred_data.json')

    # load prediction json
    all_json_info = []
    with open(input_json, 'r') as fp:
        all_json_info = json.load(fp)['data']

    # aggregate results
    padding = ['2'] * (CLIP_FRAMES // 2)
    all_json_info = sorted(all_json_info, key=itemgetter('id'))
    groups = [OrderedDict([('path', k),
                           ('num_frames', 0),
                           ('framerate', i['framerate']),
                           # recover original bit stream
                           ('bit_stream', ''.join(val if idx == 0 else val[-1]\
                               for idx, val in enumerate([i['bit_stream'] for i in g]))),
                           # ground truth bit stream
                           ('ground_truth_bit_stream', np.array(padding + [str(int(i['label'])) for i in g] + padding)),
                           ('ground_truth_bit_stream_fixed', None),
                           # predicted bit stream
                           ('predicted_bit_stream', np.array(padding + [str(int(i['pred_label'])) for i in g] + padding)),
                           ('predicted_bit_stream_fixed', None)
                          ])
              for k, g, in ((k, list(g)) for k, g in groupby(all_json_info, itemgetter('path')))]
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # modify bit streams
    ds_path = ''
    file_count = 0
    replace = '0' * SILENT_CONSECUTIVE_FRAMES
    for idx, item in enumerate(groups):
        file_count += 1
        cur_item = item
        if ds_path == '':
            ds_path = cur_item['path']
        else:
            ds_path = find_common_path(ds_path, cur_item['path'])
        cur_item['num_frames'] = len(cur_item['bit_stream'])

        # modify ground truth bit stream
        gt_bs = cur_item['ground_truth_bit_stream']
        cur_item['ground_truth_bit_stream'] = ''.join(gt_bs)
        overwritten_gt_bs = []
        for i, l in enumerate(gt_bs):
            if l == '0':
                overwritten_gt_bs += list(range(i-SILENT_CONSECUTIVE_FRAMES//2, i+SILENT_CONSECUTIVE_FRAMES//2+1))
                # gt_bs = gt_bs[:i-SILENT_CONSECUTIVE_FRAMES//2] + replace + gt_bs[i+SILENT_CONSECUTIVE_FRAMES//2+1:]
        if overwritten_gt_bs:
            overwritten_gt_bs = np.array(overwritten_gt_bs)
            gt_bs[overwritten_gt_bs] = '0'
        gt_bs = ''.join(gt_bs)
        cur_item['ground_truth_bit_stream_fixed'] = gt_bs

        # modify predicted bit bit stream
        p_bs = cur_item['predicted_bit_stream']
        cur_item['predicted_bit_stream'] = ''.join(p_bs)
        overwritten_p_bs = []
        for i, l in enumerate(p_bs):
            if l == '0':
                overwritten_p_bs += list(range(i-SILENT_CONSECUTIVE_FRAMES//2, i+SILENT_CONSECUTIVE_FRAMES//2+1))
                # p_bs = p_bs[:i-SILENT_CONSECUTIVE_FRAMES//2] + replace + p_bs[i+SILENT_CONSECUTIVE_FRAMES//2+1:]
        if overwritten_p_bs:
            overwritten_p_bs = np.array(overwritten_p_bs)
            p_bs[overwritten_p_bs] = '0'
        p_bs = ''.join(p_bs)
        cur_item['predicted_bit_stream_fixed'] = p_bs

        groups[idx] = cur_item
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # save as json object
    hierarchy = OrderedDict([
        ('dataset_path', ds_path),
        ('num_videos', file_count),
        ('num_frames_per_data', CLIP_FRAMES),
        ('num_center_silent_consecutive_frames', SILENT_CONSECUTIVE_FRAMES),
        ('files', groups)
    ])

    # write json
    with open(output_json, 'wb') as f:
        json_str = json.dumps(hierarchy, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())
    print('Done')


def create_data_from_prediction_newloss(input_json, output_json=None, suffix=None):
    if output_json is None:
        output_json = os.path.join(os.path.abspath(os.path.join(input_json, os.pardir)), 'pred_data.json')
        if suffix is not None and suffix != "":
            output_json = output_json.split('.json')[0] + '{}.json'.format(suffix)

    # load prediction json
    all_json_info = []
    data_total_frames = 0
    data_center_frames = 0
    min_silent_interval = 10
    sigmoid_threshold = 0.5
    with open(input_json, 'r') as fp:
        json_obj = json.load(fp)
        all_json_info = json_obj['data']
        data_total_frames = json_obj['data_total_frames']
        # data_center_frames = json_obj['data_center_frames']
        data_center_frames = 1
        min_silent_interval = json_obj['min_silent_interval']
        sigmoid_threshold = json_obj['sigmoid_threshold']

    # aggregate results
    padding = ['2'] * (data_total_frames // 2)
    all_json_info = sorted(all_json_info, key=itemgetter('id'))
    groups = [OrderedDict([('path', k),
                           ('num_frames', 0),
                           ('framerate', i['framerate']),
                           # recover original bit stream
                           ('bit_stream', ''.join(val if idx == 0 else val[-1]\
                               for idx, val in enumerate([i['bit_stream'] for i in g]))),
                           # ground truth bit stream
                           ('ground_truth_bit_stream', np.array(padding + [str(int(i['label'])) for i in g] + padding)),
                           ('ground_truth_bit_stream_fixed', None),
                           # predicted bit stream
                           ('predicted_bit_stream', np.array(padding + [str(int(i['pred_label'])) for i in g] + padding)),
                           ('predicted_bit_stream_fixed', None)
                          ])
              for k, g, in ((k, list(g)) for k, g in groupby(all_json_info, itemgetter('path')))]
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # modify bit streams
    ds_path = ''
    file_count = 0
    replace = '0' * data_center_frames
    for idx, item in enumerate(groups):
        file_count += 1
        cur_item = item
        if ds_path == '':
            ds_path = cur_item['path']
        else:
            ds_path = find_common_path(ds_path, cur_item['path'])
        cur_item['num_frames'] = len(cur_item['bit_stream'])

        # modify ground truth bit stream
        gt_bs = cur_item['ground_truth_bit_stream']
        cur_item['ground_truth_bit_stream'] = ''.join(gt_bs)
        overwritten_gt_bs = []
        for i, l in enumerate(gt_bs):
            if l == '0':
                overwritten_gt_bs += list(range(i-data_center_frames//2, i+data_center_frames//2+1))
                # gt_bs = gt_bs[:i-data_center_frames//2] + replace + gt_bs[i+data_center_frames//2+1:]
        if overwritten_gt_bs:
            overwritten_gt_bs = np.array(overwritten_gt_bs)
            gt_bs[overwritten_gt_bs] = '0'
        gt_bs = ''.join(gt_bs)
        cur_item['ground_truth_bit_stream_fixed'] = gt_bs

        # filter out short silent intervals in ground truth bit stream
        gt_bs_fixed = cur_item['ground_truth_bit_stream_fixed']
        gt_bs_fixed = np.array(list(gt_bs_fixed))
        cur_item['ground_truth_bit_stream_fixed'] = ''.join(gt_bs_fixed)
        gt_bs_fixed_groups = [(k, len(list(g))) for k, g in groupby(gt_bs_fixed)]
        gt_bs_fixed_groups_with_idx = []
        g_idx = 0
        for g in gt_bs_fixed_groups:
            gt_bs_fixed_groups_with_idx.append((*g, g_idx))
            g_idx += g[1]
        overwritten_gt_bs = []
        for g in gt_bs_fixed_groups_with_idx:
            if g[0] == '0' and g[1] < min_silent_interval:
                overwritten_gt_bs += list(range(g[2], g[2]+g[1]))
        if overwritten_gt_bs:
            overwritten_gt_bs = np.array(overwritten_gt_bs)
            gt_bs_fixed[overwritten_gt_bs] = '1'
        gt_bs_fixed = ''.join(gt_bs_fixed)
        cur_item['ground_truth_bit_stream_fixed'] = gt_bs_fixed

        # modify predicted bit bit stream
        p_bs = cur_item['predicted_bit_stream']
        cur_item['predicted_bit_stream'] = ''.join(p_bs)
        overwritten_p_bs = []
        for i, l in enumerate(p_bs):
            if l == '0':
                overwritten_p_bs += list(range(i-data_center_frames//2, i+data_center_frames//2+1))
                # p_bs = p_bs[:i-data_center_frames//2] + replace + p_bs[i+data_center_frames//2+1:]
        if overwritten_p_bs:
            overwritten_p_bs = np.array(overwritten_p_bs)
            p_bs[overwritten_p_bs] = '0'
        p_bs = ''.join(p_bs)
        cur_item['predicted_bit_stream_fixed'] = p_bs

        # filter out short silent intervals in ground truth bit stream
        p_bs_fixed = cur_item['predicted_bit_stream_fixed']
        p_bs_fixed = np.array(list(p_bs_fixed))
        cur_item['predicted_bit_stream_fixed'] = ''.join(p_bs_fixed)
        p_bs_fixed_groups = [(k, len(list(g))) for k, g in groupby(p_bs_fixed)]
        p_bs_fixed_groups_with_idx = []
        g_idx = 0
        for g in p_bs_fixed_groups:
            p_bs_fixed_groups_with_idx.append((*g, g_idx))
            g_idx += g[1]
        overwritten_p_bs = []
        for g in p_bs_fixed_groups_with_idx:
            if g[0] == '0' and g[1] < min_silent_interval:
                overwritten_p_bs += list(range(g[2], g[2]+g[1]))
        if overwritten_p_bs:
            overwritten_p_bs = np.array(overwritten_p_bs)
            p_bs_fixed[overwritten_p_bs] = '1'
        p_bs_fixed = ''.join(p_bs_fixed)
        cur_item['predicted_bit_stream_fixed'] = p_bs_fixed

        groups[idx] = cur_item
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # save as json object
    hierarchy = OrderedDict([
        ('dataset_path', ds_path),
        ('num_videos', file_count),
        ('sigmoid_threshold', sigmoid_threshold), 
        ('min_silent_interval', min_silent_interval),
        ('data_total_frames', data_total_frames),
        ('data_center_frames', data_center_frames),
        ('files', groups)
    ])

    # write json
    with open(output_json, 'wb') as f:
        json_str = json.dumps(hierarchy, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())
    print('Done')


def pred_labels_to_actual_bitstream(arr, k=1, weights=None, threshold=0.5):
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert np.array_equal(arr, arr.astype(bool))
    if weights is not None:
        assert len(weights) == len(arr)

    # labels matrix
    m_labels = np.zeros((len(arr), len(arr)+k-1), dtype=int)
    for row_idx, row in enumerate(m_labels):
        row[row_idx:row_idx+k] = arr[row_idx]
    # print('predicted labels:', np.transpose(m_labels))

    # weights matrix
    m_weights = np.zeros((len(arr), len(arr)+k-1), dtype=float)
    for row_idx, row in enumerate(m_weights):
        if weights is None:
            # row[row_idx:row_idx+k] = gauss_weights(k)
            row[row_idx:row_idx+k] = uniform_weights(k)
        else:
            row[row_idx:row_idx+k] = np.full(k, weights[row_idx], np.float)
    # print('corr. weights:', m_weights)

    # normalize weights column-wise
    # m_weights /= m_weights.sum(axis=0, keepdims=True)[:]
    # print('weights normalized:', np.transpose(m_weights))

    # column-wise dot product
    scores = np.einsum('ij,ij->j', m_labels, m_weights)[k-1:-k+1]
    # print('scores:', scores)

    min_score = np.min(scores)
    max_score = np.max(scores)
    if min_score < 0 or max_score > 1:
        minmax_scale(scores, feature_range=(0, 1), copy=False)

    # convert to binary array
    bs = np.where(scores >= threshold, 1, 0)

    bs = np.pad(bs, k-1, mode='constant', constant_values=2)
    # print(bs)
    
    # return np.rollaxis(np.dstack((m_labels, m_weights)), 2, 0)
    # return np.swapaxes(np.dstack((m_labels, m_weights)), 2, 0)    # already transposed
    return ''.join(map(str, bs))


def pred_labels_to_floatstream(arr, k=1, weights=None, toprint=False):
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    assert np.array_equal(arr, arr.astype(bool))
    if weights is not None:
        assert len(weights) == len(arr)

    # convert input predicted labels into -1 (silent) and 1 (non-silent) for easy computation
    arr_copy = np.copy(arr)
    arr_copy[arr == 0] = -1.0

    # labels matrix
    m_labels = np.zeros((len(arr_copy), len(arr_copy)+k-1), dtype=int)
    for row_idx, row in enumerate(m_labels):
        row[row_idx:row_idx+k] = arr_copy[row_idx]
    if toprint:
        print('predicted labels:', np.transpose(m_labels))

    # weights matrix
    m_weights = np.zeros((len(arr_copy), len(arr_copy)+k-1), dtype=float)
    for row_idx, row in enumerate(m_weights):
        if weights is None:
            # row[row_idx:row_idx+k] = gauss_weights(k)
            row[row_idx:row_idx+k] = uniform_weights(k)
        else:
            row[row_idx:row_idx+k] = np.full(k, weights[row_idx], np.float)
    # print('corr. weights:', m_weights)

    # normalize weights column-wise
    # DON'T NORMALIZE HERE (PER FRAME)
    # m_weights /= m_weights.sum(axis=0, keepdims=True)[:]
    # if toprint:
    #     print('weights normalized:', np.transpose(m_weights))

    # column-wise dot product
    scores = np.einsum('ij,ij->j', m_labels, m_weights)[k-1:-k+1]
    if toprint:
        print('scores:', scores)

    # NORMALIZE HERE (PER VIDEO)
    min_score = np.min(scores)
    max_score = np.max(scores)
    if min_score < -1 or max_score > 1:
        minmax_scale(scores, feature_range=(-1, 1), copy=False)
    if toprint:
        print('scaled scores:', scores)

    # Trim first k-1 and last k-1 frames
    # to get scores that were covered k times
    # scores = np.pad(scores[k-1:-k+1], k//2, mode='constant', constant_values=0)

    return scores


def create_data_from_prediction_newtarget_bceloss(input_json, output_json=None, suffix=None, noise_snr=None, save_results=True, overwrite=False, need_filling=False, clean_audio=True):
    print(input_json)
    VOTING = '_voting'
    if output_json is None:
        output_json = os.path.join(os.path.abspath(os.path.join(input_json, os.pardir)), 'pred_data.json')
        if suffix is not None and suffix != "":
            output_json = output_json.split('.json')[0] + '{}.json'.format(suffix)
    nsuffix = convert_snr_to_suffix2(noise_snr)
    output_json = output_json.split('.json')[0] + '{}.json'.format(nsuffix + VOTING)

    # load prediction json
    all_json_info = []
    data_total_frames = 0
    data_center_frames = 0
    sigmoid_threshold = 0.5
    with open(input_json, 'r') as fp:
        json_obj = json.load(fp)
        all_json_info = json_obj['data']
        data_total_frames = json_obj['data_total_frames']
        data_center_frames = json_obj['data_center_frames']
        sigmoid_threshold = json_obj['sigmoid_threshold']

    # aggregate results
    # padding = ['2'] * (data_total_frames // 2)
    all_json_info = sorted(all_json_info, key=itemgetter('id'))

    # fill gap
    if need_filling:
        # print([(i['id'], i['frame_start_idx'], i['num_frames']) for i in all_json_info])
        # print(len(all_json_info))
        filled_all_json_info = []
        for index, kg in enumerate(((k, list(g)) for k, g in groupby(all_json_info, itemgetter('path')))):
            k = kg[0]
            g = kg[1]
            # print(k, g)
            # print(k)
            # print(g)

            item_counter = 0

            existing = [(i['id'], i['frame_start_idx']) for i in g]
            full = [i for i in range(0, g[0]['num_frames']-data_total_frames+1)]
            # print(existing)
            # print(full)
            # missing = [x for x in set(full) if x not in existing]
            missing = []
            id_counter = 0
            for x in set(full):
                if x not in (x_i[1] for x_i in existing):
                    missing.append((id_counter, x))
                    g.insert(x, OrderedDict([
                            ('id', id_counter),
                            ('path', k),
                            ('full_bit_stream', g[0]['full_bit_stream']),
                            ('num_frames', g[0]['num_frames']),
                            ('framerate', g[0]['framerate']),
                            ('frame_start_idx', x),
                            ('label', 1.0),
                            ('pred_label', 1.0),
                            ('match', True),
                            ('confidence', 1.0)
                        ]))
                else:
                    id_counter += 1
            # print(missing)

            assert len(existing) + len(missing) == len(full)
            filled_all_json_info += g           

            # break

        # print(len(filled_all_json_info))
        # print(sum([g[0]['num_frames'] for k, g in ((k, list(g)) for k, g in groupby(all_json_info, itemgetter('path')))]))
        # exit()

    else:
        filled_all_json_info = all_json_info

    groups = [OrderedDict([('path', k),
                           ('num_frames', g[0]['num_frames']),
                           ('framerate', g[0]['framerate']),
                           ('audio_sample_rate', g[0]['audio_sample_rate']),
                           ('audio_samples', g[0]['audio_samples']),
                           ('duration', g[0]['duration']),
                           # recover original bit stream
                        #    ('bit_stream', ''.join(val if idx == 0 else val[-1]\
                        #        for idx, val in enumerate([i['bit_stream'] for i in g]))),
                           ('bit_stream', g[0]['full_bit_stream']),
                           # ground truth bit stream
                        #    ('ground_truth_bit_stream', ''.join([str(int(i['label'])) for i in g])),
                           ('ground_truth_bit_stream', ''.join([str(int(element)) for i in g for element in i['label']])),
                        #    ('ground_truth_bit_stream_fixed', None),
                           # predicted bit stream
                        #    ('predicted_bit_stream', np.array([int(i['pred_label']) for i in g])),
                           ('predicted_bit_stream', np.array([int(element) for i in g for element in i['pred_label']])),
                        #    ('predicted_bit_stream_fixed', None)
                        #    ('confidences', np.array([float(i['confidence']) for i in g])),
                           ('confidences', np.array([float(element) for i in g for element in i['confidence']])),
                           ('recovered_prediction', None),
                           ('overlay_original', None),
                           ('overlay_predicted', None)
                          ])
              for k, g, in ((k, list(g)) for k, g in groupby(filled_all_json_info, itemgetter('path')))]
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # modify bit streams
    ds_path = ''
    file_count = 0
    # replace = '0' * data_center_frames

    labels = []
    pred_labels = []
    all_scores_uniform_weights = []
    all_scores_norm_diff_weights = []
    all_scores_squared_norm_diff_weights = []

    for idx, item in enumerate(tqdm(groups)):
        file_count += 1
        cur_item = item
        if ds_path == '':
            ds_path = cur_item['path']
        else:
            ds_path = find_common_path(ds_path, cur_item['path'])
        cur_item['num_frames'] = len(cur_item['bit_stream'])
        cur_item['recovered_prediction'] = pred_labels_to_actual_bitstream(cur_item['predicted_bit_stream'], k=data_total_frames, threshold=0.5)
        # cur_item['predicted_bit_stream'] = ''.join(map(str, cur_item['predicted_bit_stream']))

        # Trim first 'data_total_frames-1' and last 'data_total_frames-1' frames
        # to get scores that were covered 'data_total_frames' times
        labels += [int(s) for s in cur_item['bit_stream']][data_total_frames-1:-data_total_frames+1]
        pred_labels += [int(s) for s in cur_item['recovered_prediction']][data_total_frames-1:-data_total_frames+1]

        # Process the raw predicted bit streams from network and confidence scores
        # 1. Same as how "recovered_prediction" is generated - uniform weights
        # 2. Normalized difference as weights
        # 3. Squared normalized difference as weights
        # 4. Raw confidence scores
        # pred_labels_to_floatstream(cur_item['predicted_bit_stream'], k=data_total_frames, weights=cur_item['confidences']),
        # pred_labels_to_floatstream(cur_item['predicted_bit_stream'], k=data_total_frames, weights=np.square(cur_item['confidences'])),
        # pred_labels_to_floatstream(cur_item['predicted_bit_stream'], k=data_total_frames, weights=np.power(cur_item['confidences'], 3))
        scores_uniform_weights = pred_labels_to_floatstream(cur_item['predicted_bit_stream'], k=data_total_frames, weights=None)
        scores_norm_diff_weights = pred_labels_to_floatstream(cur_item['predicted_bit_stream'], k=data_total_frames,\
            weights=[(i-sigmoid_threshold)/(1-sigmoid_threshold)\
                if i > sigmoid_threshold else (sigmoid_threshold-i)/sigmoid_threshold for i in cur_item['confidences']])
        scores_squared_norm_diff_weights = pred_labels_to_floatstream(cur_item['predicted_bit_stream'], k=data_total_frames,\
            weights=np.square([(i-sigmoid_threshold)/(1-sigmoid_threshold)\
                if i > sigmoid_threshold else (sigmoid_threshold-i)/sigmoid_threshold for i in cur_item['confidences']]))
        # np.concatenate((np.full(CLIP_FRAMES//2, 0, np.float), cur_item['confidences'], np.full(CLIP_FRAMES//2, 0, np.float)))
        scores_raw_confidence = cur_item['confidences']
        # print(len(scores_raw_confidence))
        all_scores_uniform_weights += list(1-(scores_uniform_weights+1)/2)
        # print(len(all_scores_uniform_weights))
        all_scores_norm_diff_weights += list(1-(scores_norm_diff_weights+1)/2)
        # print(len(all_scores_norm_diff_weights))
        all_scores_squared_norm_diff_weights += list(1-(scores_squared_norm_diff_weights+1)/2)
        # print(len(all_scores_squared_norm_diff_weights))

        # save results
        if save_results:
            # Preparation
            save_dir = os.path.join(os.path.abspath(os.path.join(input_json, os.pardir)), 'recovered' + suffix + nsuffix + VOTING)
            ensure_dir(save_dir)
            wav_path = cur_item['path'].split('.mp4')[0] if len(cur_item['path'].split('.mp4')) == 1 else cur_item['path'].split('.mp4')[0] + '.wav'
            snd, _ = librosa.load(wav_path, sr=DATA_REQUIRED_SR)
            filename = os.path.basename(cur_item['path']).split('.mp4')[0]

            # Draw float streams overlay figure
            wav_float_path = os.path.join(save_dir, filename + '_overlay_predicted_floats.png')
            cur_item['overlay_predicted'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(wav_float_path, os.pardir))), os.path.basename(wav_float_path))
            if (not os.path.exists(wav_float_path)) or overwrite:
                if data_total_frames % 2 == 1:
                    float_streams = np.vstack((
                        np.pad(scores_uniform_weights, data_total_frames-1, mode='constant', constant_values=0),
                        np.pad(scores_norm_diff_weights, data_total_frames-1, mode='constant', constant_values=0),
                        np.pad(scores_squared_norm_diff_weights, data_total_frames-1, mode='constant', constant_values=0),
                        np.pad(scores_raw_confidence, data_total_frames//2, mode='constant', constant_values=0)
                    ))
                else:
                    float_streams = np.vstack((
                        np.pad(scores_uniform_weights, data_total_frames-1, mode='constant', constant_values=0),
                        np.pad(scores_norm_diff_weights, data_total_frames-1, mode='constant', constant_values=0),
                        np.pad(scores_squared_norm_diff_weights, data_total_frames-1, mode='constant', constant_values=0),
                        np.pad(scores_raw_confidence, data_total_frames//2, mode='constant', constant_values=0)[:-1]
                    ))

                # fs_labels = ['uniform weights', 'confidences as weights', 'squared confidences as weights', 'norm. diff. as weights', 'norm. squared diff. as weights']
                fs_labels = ['uniform weights', 'norm. diff. as weights', 'squared norm. diff. as weights', 'raw confidence score']

                # add noise to clean audio
                if clean_audio:
                    noise_json_path = os.path.join(get_parent_dir(output_json), 'noise' + nsuffix, nsuffix[1:] + '.json')
                    snrs = []
                    noise_files = OrderedDict()
                    with open(noise_json_path, 'r') as fpn:
                        noise_json_obj = json.load(fpn)
                        snrs = noise_json_obj['snrs']
                        noise_files = noise_json_obj['files']
                    # noise_name = noise_files[os.path.basename(cur_item['path'])]['noise']
                    noise_item = [x for x in noise_files if x['path'] == cur_item['path']][0]
                    noise_name = noise_item['noise']
                    noise_path = os.path.join(get_parent_dir(output_json), 'noise' + nsuffix, noise_name)
                    noise, _ = librosa.load(noise_path, sr=DATA_REQUIRED_SR)[:len(snd)]
                    # snr = noise_files[os.path.basename(cur_item['path'])]['snr']
                    snr = noise_item['snr']
                    # print('snd len: ', snd.shape)
                    # print('noise len: ', noise.shape)
                    # print('snr: ', snr)
                    audio_mixed, audio_clean, audio_full_noise = add_noise_to_audio(snd, noise, snr=snr, start_pos=0, norm=0.5)
                    # audio_mixed = librosa.util.normalize(audio_mixed)
                    plot_wav_floatstreams_overlay(audio_mixed, DATA_REQUIRED_SR, float_streams, labels=fs_labels, plot_path=wav_float_path, suppress_stdout=True)

                    audio_mixed_path = os.path.join(save_dir, filename + '_mixed.wav')
                    librosa.output.write_wav(audio_mixed_path, audio_mixed, DATA_REQUIRED_SR)
                    cur_item['mixed_audio'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_mixed_path, os.pardir))), os.path.basename(audio_mixed_path))

                    audio_clean_path = os.path.join(save_dir, filename + '_clean.wav')
                    librosa.output.write_wav(audio_clean_path, audio_clean, DATA_REQUIRED_SR)
                    cur_item['clean_audio'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_clean_path, os.pardir))), os.path.basename(audio_clean_path))

                    audio_full_noise_path = os.path.join(save_dir, filename + '_full_noise.wav')
                    librosa.output.write_wav(audio_full_noise_path, audio_full_noise[0], DATA_REQUIRED_SR)
                    cur_item['full_noise'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_full_noise_path, os.pardir))), os.path.basename(audio_full_noise_path))

                    cur_item['audio_path'] = audio_clean_path
                else:
                    # audio_mixed = librosa.util.normalize(snd)
                    # scale = np.max(np.abs(audio_mixed)) / 0.5
                    # audio_mixed = audio_mixed / scale
                    # audio_clean = audio_mixed
                    audio_mixed = snd
                    audio_clean = snd
                    plot_wav_floatstreams_overlay(audio_mixed, DATA_REQUIRED_SR, float_streams, labels=fs_labels, plot_path=wav_float_path, suppress_stdout=True)
                    audio_mixed_path = os.path.join(save_dir, filename + '_mixed.wav')
                    librosa.output.write_wav(audio_mixed_path, audio_mixed, DATA_REQUIRED_SR)
                    cur_item['mixed_audio'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_mixed_path, os.pardir))), os.path.basename(audio_mixed_path))

            # Draw waveform and bitstream overlay
            wav_bit_path = os.path.join(save_dir, filename + '_overlay_original.png')
            cur_item['overlay_original'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(wav_bit_path, os.pardir))), os.path.basename(wav_bit_path))
            if (not os.path.exists(wav_bit_path)) or overwrite:
                plot_wav_bitstream_overlay(audio_clean, DATA_REQUIRED_SR, cur_item['bit_stream'], wav_bit_path, True)

        cur_item['predicted_bit_stream'] = ''.join(map(str, cur_item['predicted_bit_stream']))
        del cur_item['confidences']

        groups[idx] = cur_item
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # save as json object
    hierarchy = OrderedDict([
        ('dataset_path', ds_path),
        ('num_videos', file_count),
        ('data_total_frames', data_total_frames),
        ('data_center_frames', data_center_frames),
        ('sigmoid_threshold', sigmoid_threshold),
        ('snr', noise_snr),
        ('prediction_statistics', show_metrics(labels, pred_labels)),
        ('files', groups)
    ])

    # Draw precision-recall curve
    # print('AP_scores_uniform_weights:\t\t', average_precision_score(labels, all_scores_uniform_weights, pos_label=0))
    # print('AP_scores_norm_diff_weights:\t\t', average_precision_score(labels, all_scores_norm_diff_weights, pos_label=0))
    # print('AP_scores_squared_norm_diff_weights:\t', average_precision_score(labels, all_scores_squared_norm_diff_weights, pos_label=0))
    
    p1, r1, _ = precision_recall_curve(labels, all_scores_uniform_weights, pos_label=0)
    p2, r2, _ = precision_recall_curve(labels, all_scores_norm_diff_weights, pos_label=0)
    p3, r3, _ = precision_recall_curve(labels, all_scores_squared_norm_diff_weights, pos_label=0)

    auc1 = auc(r1, p1)
    auc2 = auc(r2, p2)
    auc3 = auc(r3, p3)
    # print('auc_scores_uniform_weights:\t\t', auc1)
    # print('auc_scores_norm_diff_weights:\t\t', auc2)
    # print('auc_scores_squared_norm_diff_weights:\t', auc3)

    # plot the calculated precision and recall
    p0 = [value for key, value in hierarchy['prediction_statistics'].items() if 'precision' in key.lower()][0]
    r0 = [value for key, value in hierarchy['prediction_statistics'].items() if 'recall' in key.lower()][0]
    plt.plot([0, 1], [p0, p0], linestyle='--', color='grey')
    plt.plot([r0, r0], [0, 1], linestyle='--', color='grey')

    plt.plot(r1, p1, label='uniform weights; AP=%.3f' % auc1)
    plt.plot(r2, p2, label='norm. diff. as weights; AP=%.3f' % auc2)
    plt.plot(r3, p3, label='squared norm. diff. as weights; AP=%.3f' % auc3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.axis('scaled')
    plt.show()
    output_pr = os.path.join(os.path.abspath(os.path.join(output_json, os.pardir)), 'pr.png')
    if suffix is not None and suffix != "":
        output_pr = output_pr.split('.png')[0] + '{}.png'.format(suffix)
    output_pr = output_pr.split('.png')[0] + '{}.png'.format(nsuffix + VOTING)
    plt.savefig(output_pr)
    hierarchy['prediction_statistics']['pr_curve'] = os.path.basename(output_pr)

    # labels_all = [item['label'] for item in stat]
    # pred_labels_all = [item['pred_label'] for item in stat]

    # print_dictionary(hierarchy, sep='\n')
    # exit()

    # write json
    with open(output_json, 'wb') as f:
        json_str = json.dumps(hierarchy, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())
    print('Done')


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


def create_data_from_prediction_newtarget_bceloss_no_voting(input_json, output_json=None, suffix=None, noise_snr=None, save_results=True, save_extra=False, overwrite=False, clean_audio=True):
    print(input_json)
    # VOTING = '_no_voting'
    VOTING = ''
    if output_json is None:
        output_json = os.path.join(os.path.abspath(os.path.join(input_json, os.pardir)), 'pred_data.json')
        if suffix is not None and suffix != "":
            output_json = output_json.split('.json')[0] + '{}.json'.format(suffix)
    nsuffix = convert_snr_to_suffix2(noise_snr)
    output_json = output_json.split('.json')[0] + '{}.json'.format(nsuffix + VOTING)

    # load prediction json
    all_json_info = []
    data_total_frames = 0
    data_center_frames = 0
    sigmoid_threshold = 0.5
    with open(input_json, 'r') as fp:
        json_obj = json.load(fp)
        all_json_info = json_obj['data']
        data_total_frames = json_obj['data_total_frames']
        data_center_frames = json_obj['data_center_frames']
        sigmoid_threshold = json_obj['sigmoid_threshold']

    # aggregate results
    # padding = ['2'] * (data_total_frames // 2)
    all_json_info = sorted(all_json_info, key=itemgetter('id'))

    # fill gap
    filled_all_json_info = all_json_info

    groups = [OrderedDict([('path', k),
                           ('num_frames', g[0]['num_frames']),
                           ('framerate', g[0]['framerate']),
                           ('audio_sample_rate', g[0]['audio_sample_rate']),
                           ('audio_samples', g[0]['audio_samples']),
                           ('duration', g[0]['duration']),
                           # recover original bit stream
                        #    ('bit_stream', ''.join(val if idx == 0 else val[-1]\
                        #        for idx, val in enumerate([i['bit_stream'] for i in g]))),
                           ('bit_stream', g[0]['full_bit_stream']),
                           # ground truth bit stream
                        #    ('ground_truth_bit_stream', ''.join([str(int(i['label'])) for i in g])),
                           ('ground_truth_bit_stream', ''.join([str(int(element)) for i in g for element in i['label']])),
                        #    ('ground_truth_bit_stream_fixed', None),
                           # predicted bit stream
                        #    ('predicted_bit_stream', np.array([int(i['pred_label']) for i in g])),
                           ('predicted_bit_stream', np.array([int(element) for i in g for element in i['pred_label']])),
                        #    ('predicted_bit_stream_fixed', None)
                        #    ('confidences', np.array([float(i['confidence']) for i in g])),
                           ('confidences', np.array([float(element) for i in g for element in i['confidence']])),
                           ('recovered_prediction', None),
                           ('overlay_original', None),
                           ('overlay_predicted', None)
                          ])
              for k, g, in ((k, list(g)) for k, g in groupby(filled_all_json_info, itemgetter('path')))]
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # modify bit streams
    ds_path = ''
    file_count = 0
    # replace = '0' * data_center_frames

    labels = []
    pred_labels = []
    all_scores_uniform_weights = []

    for idx, item in enumerate(tqdm(groups)):
        file_count += 1
        cur_item = item
        if ds_path == '':
            ds_path = cur_item['path']
        else:
            ds_path = find_common_path(ds_path, cur_item['path'])
        cur_item['num_frames'] = len(cur_item['bit_stream'])
        # cur_item['recovered_prediction'] = pred_labels_to_actual_bitstream(cur_item['predicted_bit_stream'], k=data_total_frames, threshold=0.5)
        cur_item['recovered_prediction'] = ''.join(map(str, cur_item['predicted_bit_stream']))

        labels += [int(s) for s in cur_item['bit_stream']]
        pred_labels += [int(s) for s in cur_item['recovered_prediction']]

        scores_uniform_weights = np.array(cur_item['confidences'] * 2 - 1)

        scores_raw_confidence = cur_item['confidences']
        # print(len(scores_raw_confidence))
        all_scores_uniform_weights += list(1-(scores_uniform_weights+1)/2)
        # print(len(all_scores_uniform_weights))

        # save results
        if save_results:
            # Preparation
            save_dir = os.path.join(os.path.abspath(os.path.join(input_json, os.pardir)), 'recovered' + suffix + nsuffix + VOTING)
            ensure_dir(save_dir)
            wav_path = cur_item['path'].split('.mp4')[0] if len(cur_item['path'].split('.mp4')) == 1 else cur_item['path'].split('.mp4')[0] + '.wav'
            snd, _ = librosa.load(wav_path, sr=DATA_REQUIRED_SR)
            filename = os.path.basename(cur_item['path']).split('.mp4')[0]

            # Draw float streams overlay figure
            if save_extra:
                wav_float_path = os.path.join(save_dir, filename + '_overlay_predicted_floats.png')
                cur_item['overlay_predicted'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(wav_float_path, os.pardir))), os.path.basename(wav_float_path))
                float_streams = np.vstack((
                    scores_uniform_weights,
                    # scores_norm_diff_weights,
                    # scores_squared_norm_diff_weights,
                    scores_raw_confidence
                ))

            fs_labels = ['scaled confidence score', 'raw confidence score']

            # add noise to clean audio
            if clean_audio:
                noise_json_path = os.path.join(get_parent_dir(output_json), 'noise' + nsuffix, nsuffix[1:] + '.json')
                snrs = []
                noise_files = OrderedDict()
                with open(noise_json_path, 'r') as fpn:
                    noise_json_obj = json.load(fpn)
                    snrs = noise_json_obj['snrs']
                    noise_files = noise_json_obj['files']
                noise_name = noise_files[os.path.basename(cur_item['path'])]['noise']
                # noise_item = [x for x in noise_files if x['path'] == cur_item['path']][0]
                # noise_name = noise_item['noise']
                noise_path = os.path.join(get_parent_dir(output_json), 'noise' + nsuffix, noise_name)
                noise, _ = librosa.load(noise_path, sr=DATA_REQUIRED_SR)[:len(snd)]
                snr = noise_files[os.path.basename(cur_item['path'])]['snr']
                # snr = noise_item['snr']
                # print('snd len: ', snd.shape)
                # print('noise len: ', noise.shape)
                # print('snr: ', snr)
                audio_mixed, audio_clean, audio_full_noise = add_noise_to_audio(snd, noise, snr=snr, start_pos=0, norm=0.5)
                # audio_mixed = librosa.util.normalize(audio_mixed)
                if save_extra:
                    if (not os.path.exists(wav_float_path)) or overwrite:
                        plot_wav_floatstreams_overlay(audio_mixed, DATA_REQUIRED_SR, float_streams, labels=fs_labels, plot_path=wav_float_path, suppress_stdout=True)

                audio_mixed_path = os.path.join(save_dir, filename + '_mixed.wav')
                librosa.output.write_wav(audio_mixed_path, audio_mixed, DATA_REQUIRED_SR)
                cur_item['mixed_audio'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_mixed_path, os.pardir))), os.path.basename(audio_mixed_path))

                audio_clean_path = os.path.join(save_dir, filename + '_clean.wav')
                librosa.output.write_wav(audio_clean_path, audio_clean, DATA_REQUIRED_SR)
                cur_item['clean_audio'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_clean_path, os.pardir))), os.path.basename(audio_clean_path))

                audio_full_noise_path = os.path.join(save_dir, filename + '_full_noise.wav')
                librosa.output.write_wav(audio_full_noise_path, audio_full_noise[0], DATA_REQUIRED_SR)
                cur_item['full_noise'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_full_noise_path, os.pardir))), os.path.basename(audio_full_noise_path))

                cur_item['audio_path'] = audio_clean_path
            else:
                # audio_mixed = librosa.util.normalize(snd)
                # scale = np.max(np.abs(audio_mixed)) / 0.5
                # audio_mixed = audio_mixed / scale
                # audio_clean = audio_mixed
                audio_mixed = snd
                audio_clean = snd
                if save_extra:
                    if (not os.path.exists(wav_float_path)) or overwrite:
                        plot_wav_floatstreams_overlay(audio_mixed, DATA_REQUIRED_SR, float_streams, labels=fs_labels, plot_path=wav_float_path, suppress_stdout=True)
                audio_mixed_path = os.path.join(save_dir, filename + '_mixed.wav')
                librosa.output.write_wav(audio_mixed_path, audio_mixed, DATA_REQUIRED_SR)
                cur_item['mixed_audio'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(audio_mixed_path, os.pardir))), os.path.basename(audio_mixed_path))

            if save_extra:
                # Draw waveform and bitstream overlay
                wav_bit_path = os.path.join(save_dir, filename + '_overlay_original.png')
                cur_item['overlay_original'] = os.path.join(os.path.basename(os.path.abspath(os.path.join(wav_bit_path, os.pardir))), os.path.basename(wav_bit_path))
                if (not os.path.exists(wav_bit_path)) or overwrite:
                    plot_wav_bitstream_overlay(audio_clean, DATA_REQUIRED_SR, cur_item['bit_stream'], wav_bit_path, True)

        cur_item['predicted_bit_stream'] = ''.join(map(str, cur_item['predicted_bit_stream']))
        del cur_item['confidences']

        groups[idx] = cur_item
    # print_dictionary(groups[0], sep='\n', omit_keys=True)

    # save as json object
    hierarchy = OrderedDict([
        ('dataset_path', ds_path),
        ('num_videos', file_count),
        ('data_total_frames', data_total_frames),
        ('data_center_frames', data_center_frames),
        ('sigmoid_threshold', sigmoid_threshold),
        ('snr', noise_snr),
        ('prediction_statistics', show_metrics(labels, pred_labels)),
        ('files', groups)
    ])

    if save_results and save_extra:
        # Draw precision-recall curve
        # print('AP_scores_uniform_weights:\t\t', average_precision_score(labels, all_scores_uniform_weights, pos_label=0))
        # print('AP_scores_norm_diff_weights:\t\t', average_precision_score(labels, all_scores_norm_diff_weights, pos_label=0))
        # print('AP_scores_squared_norm_diff_weights:\t', average_precision_score(labels, all_scores_squared_norm_diff_weights, pos_label=0))
        
        p1, r1, _ = precision_recall_curve(labels, all_scores_uniform_weights, pos_label=0)
        # p2, r2, _ = precision_recall_curve(labels, all_scores_norm_diff_weights, pos_label=0)
        # p3, r3, _ = precision_recall_curve(labels, all_scores_squared_norm_diff_weights, pos_label=0)

        auc1 = auc(r1, p1)
        # auc2 = auc(r2, p2)
        # auc3 = auc(r3, p3)
        # print('auc_scores_uniform_weights:\t\t', auc1)
        # print('auc_scores_norm_diff_weights:\t\t', auc2)
        # print('auc_scores_squared_norm_diff_weights:\t', auc3)

        # plot the calculated precision and recall
        p0 = [value for key, value in hierarchy['prediction_statistics'].items() if 'precision' in key.lower()][0]
        r0 = [value for key, value in hierarchy['prediction_statistics'].items() if 'recall' in key.lower()][0]
        plt.plot([0, 1], [p0, p0], linestyle='--', color='grey')
        plt.plot([r0, r0], [0, 1], linestyle='--', color='grey')

        plt.plot(r1, p1, label='scaled confidence; AP=%.3f' % auc1)
        # plt.plot(r2, p2, label='norm. diff. as weights; AP=%.3f' % auc2)
        # plt.plot(r3, p3, label='squared norm. diff. as weights; AP=%.3f' % auc3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.axis('scaled')
        plt.show()
        output_pr = os.path.join(os.path.abspath(os.path.join(output_json, os.pardir)), 'pr.png')
        if suffix is not None and suffix != "":
            output_pr = output_pr.split('.png')[0] + '{}.png'.format(suffix)
        output_pr = output_pr.split('.png')[0] + '{}.png'.format(nsuffix + VOTING)
        plt.savefig(output_pr)
        hierarchy['prediction_statistics']['pr_curve'] = os.path.basename(output_pr)

        # labels_all = [item['label'] for item in stat]
        # pred_labels_all = [item['pred_label'] for item in stat]

        # print_dictionary(hierarchy, sep='\n')
        # exit()

    # write json
    with open(output_json, 'wb') as f:
        json_str = json.dumps(hierarchy, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())
    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=str, default="", required=False, help="specify new threshold")
    parser.add_argument('--snr', type=float, default=None)
    parser.add_argument('--save_results', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)
    parser.add_argument('--save_extra', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)
    parser.add_argument('--unknown_clean_signal', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)
    args = parser.parse_args()

    suffix = convert_threshold_to_suffix(args.threshold)
    input_json = 'eval_results.json'.split('.json')[0] + '{}.json'.format(suffix)
    nsuffix = convert_snr_to_suffix2(args.snr)
    input_json = input_json.split('.json')[0] + '{}.json'.format(nsuffix)
    print(input_json)
    if not args.unknown_clean_signal:
        create_data_from_prediction_newtarget_bceloss_no_voting(os.path.join(EXPERIMENT_PREDICTION_OUTPUT_DIR, input_json),\
            suffix=suffix, noise_snr=args.snr, overwrite=True, save_results=args.save_results, save_extra=args.save_extra)
    else:
        create_data_from_prediction_newtarget_bceloss_no_voting(os.path.join(EXPERIMENT_PREDICTION_OUTPUT_DIR, os.path.basename(CE_JSON).split('.json')[0], input_json),\
            suffix=suffix, noise_snr=args.snr, overwrite=True, save_results=args.save_results, save_extra=args.save_extra, clean_audio=False)

    # python3 create_data_from_pred.py --unknown_clean_signal true
