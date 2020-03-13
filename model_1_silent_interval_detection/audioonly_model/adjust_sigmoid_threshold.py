import argparse
import errno
import json
import os
import numpy as np

from collections import OrderedDict
from tqdm import tqdm

from operator import itemgetter
from itertools import groupby

from dataset import CLIP_FRAMES, SILENT_CONSECUTIVE_FRAMES
from utils import print_dictionary, find_common_path
from tools import *
from common import OUTPUT_ROOT, EXPERIMENT_DIR
from predict import EXPERIMENT_PREDICTION_OUTPUT_DIR


def adjust_sigmoid_threshold(threshold, input_json, output_json=None):
    if output_json is None:
        output_json = os.path.abspath(input_json).split('.json')[0]\
            + '_{}.json'.format(str(threshold).replace('.', '_'))

    # load prediction json
    all_json_info = []
    with open(os.path.abspath(input_json), 'r') as fp:
        all_json_info = json.load(fp)['data']

    for info in all_json_info:
        confidence = info['confidence']
        pred_label = 1.0 if confidence >= threshold else 0.0
        info['pred_label'] = pred_label
        info['match'] = (info['label'] == pred_label)

    stat_dict = OrderedDict()

    stat_dict['data_total_frames'] = CLIP_FRAMES
    stat_dict['data_center_frames'] = SILENT_CONSECUTIVE_FRAMES
    stat_dict['sigmoid_threshold'] = threshold

    labels_all = [item['label'] for item in all_json_info]
    pred_labels_all = [item['pred_label'] for item in all_json_info]
    labels_partial = [item['label'] for item in all_json_info\
        if item['bit_stream'].count('0') == len(item['bit_stream'])\
                or item['bit_stream'].count('1') > len(item['bit_stream']) - 5]
    pred_labels_partial = [item['pred_label'] for item in all_json_info\
        if item['bit_stream'].count('0') == len(item['bit_stream'])\
                or item['bit_stream'].count('1') > len(item['bit_stream']) - 5]
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

    # sort all_json_info by confidence descending order
    all_json_info = sorted(all_json_info, key=lambda x:x['confidence'], reverse=True)
    stat_dict['data'] = all_json_info

    save_stat_path = os.path.abspath(output_json)
    with open(save_stat_path, 'w') as fp:
        json.dump(stat_dict, fp, **JSON_DUMP_PARAMS)
    print('Overall results saved to: \'{}\''.format(save_stat_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float, default=0.5, required=False, help="specify new threshold")
    args = parser.parse_args()

    adjust_sigmoid_threshold(args.threshold, os.path.join(EXPERIMENT_PREDICTION_OUTPUT_DIR, 'eval_results.json'))
