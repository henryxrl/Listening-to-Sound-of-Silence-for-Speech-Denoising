import json
import os
from collections import OrderedDict

import cv2
import numpy as np
from scipy import ndimage
# from PIL import Image
from shapely.geometry import Polygon, box
from shapely.ops import cascaded_union
from tqdm import tqdm

from tools import DATA_ROOT, JSON_DUMP_PARAMS, ensure_dir, get_number_of_files, get_parent_dir, get_path_same_dir


PROTOTXT_PATH = os.path.join('face_detection_model/deploy.prototxt')
# CAFFEMODEL_PATH = os.path.join('face_detection_model/weights.caffemodel')
CAFFEMODEL_PATH = os.path.join('face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
MODEL = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)


def get_initial_bb(img_path, x, y):
    """
    Crop image using x and y coordinates
    Modified from network tools.py crop_face function
    
    """
    magic_hw_ration = 1.75
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    dists_center = {
        'l': w * x,
        'r': w * (1.0 - x),
        't': h * y,
        'b': h * (1.0 - y)
    }
    smallest = min(dists_center, key=dists_center.get)
    if smallest in ('t', 'b'):
        crop_tuple = (
            dists_center['l'] - dists_center[smallest],
            dists_center['t'] - dists_center[smallest],
            w - (dists_center['r'] - dists_center[smallest]),
            h - (dists_center['b'] - dists_center[smallest])
        )
    else:
        crop_tuple = (
            dists_center['l'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['l'] else 0,
            dists_center['t'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['t'] else 0,
            w - (dists_center['r'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['r'] else 0),
            h - (dists_center['b'] - dists_center[smallest] * magic_hw_ration\
                if dists_center[smallest] * magic_hw_ration <= dists_center['b'] else 0)
        )
    return crop_tuple


def get_iou(bb1, bb2, epsilon=1e-5):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : [x1, y1, x2, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    epsilon : (float) Small value to prevent division by zero

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] <= bb1[2]
    assert bb1[1] <= bb1[3]
    assert bb2[0] <= bb2[2]
    assert bb2[1] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area + epsilon)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou2(bb1, bb2, epsilon=1e-5):
    # print('in get_iou2')
    # print(bb1)
    # print(bb2)
    # poly_1 = Polygon(*bb1)
    poly_1 = box(*bb1)
    # print(poly_1)
    # poly_2 = Polygon(*bb2)
    poly_2 = box(*bb2)
    # print(poly_2)
    iou = poly_1.intersection(poly_2).area / (poly_1.union(poly_2).area + epsilon)
    # print(iou)
    return iou


def get_min_union_bb(bb1, bb2):
    # polygons = [box(i[0],i[1],i[2],i[3]) for i in bbox_list]
    polygons = [box(*bb1), box(*bb2)]
    return cascaded_union(polygons).bounds


def get_bb(initial_bb, detections, h, w, top_confidence=5, suppress_stdout=True):
    top_confidence = top_confidence if top_confidence > 0 else detections.shape[2]
    iou_ranks_dict = [{'iou':get_iou(initial_bb, detections[0, 0, i, 3:7] * np.array([w, h, w, h])),\
        'confidence':detections[0, 0, i, 2], 'index':i}\
            for i in range(0, detections.shape[2]) if i < top_confidence]
    if not suppress_stdout:
        print('iou_ranks_dict:', iou_ranks_dict)
    iou_ranks = [i['iou'] for i in iou_ranks_dict]
    if not suppress_stdout:
        print('get_iou ranks:', iou_ranks)
    max_iou = max(iou_ranks)
    if not suppress_stdout:
        print(max_iou)
    if max_iou != 0.0:
        index = iou_ranks.index(max_iou)
        if not suppress_stdout:
            print(index)
    else:
        iou_ranks = [get_iou2(initial_bb, detections[0, 0, i, 3:7] * np.array([w, h, w, h]))\
            for i in range(0, detections.shape[2])]
        if not suppress_stdout:
            print('get_iou2 ranks:', iou_ranks)
        max_iou = max(iou_ranks)
        if not suppress_stdout:
            print(max_iou)
        index = iou_ranks.index(max_iou)
        if not suppress_stdout:
            print(index)    
    return detections[0, 0, index, 3:7] * np.array([w, h, w, h])


def make_square_img(startX, startY, endX, endY, h, w, min_size=256, suppress_stdout=True):
    if not suppress_stdout:
        print('old: ', startX, startY, endX, endY)
        print('w: {}; h: {}'.format(w, h))
    width = endX - startX
    height = endY - startY
    centerX = startX + width // 2
    centerY = startY + height // 2
    half = max(width, height, min_size) // 2
    if not suppress_stdout:
        print(width, height, centerX, centerY, half)

    newstartX = centerX - half
    newstartY = centerY - half
    newendX = centerX + half
    newendY = centerY + half
    if not suppress_stdout:
        print('tmp: ', newstartX, newstartY, newendX, newendY)

    padding_left = 0 if newstartX >= 0 else -newstartX
    padding_top = 0 if newstartY >= 0 else -newstartY
    padding_right = 0 if newendX < w else newendX - w + 1
    padding_bottom = 0 if newendY < h else newendY - h + 1
    if not suppress_stdout:
        print('padding: ', padding_left, padding_top, padding_right, padding_bottom)

    # newstartX = newstartX if padding_left == 0 else 0
    # newstartY = newstartY if padding_top == 0 else 0
    # newendX = newendX if padding_right == 0 else w - 1
    # newendY = newendY if padding_bottom == 0 else h - 1
    if padding_left > 0:
        newstartX = 0
        newendX += padding_left
    if padding_top > 0:
        newstartY = 0
        newendY += padding_top

    return (newstartX, newstartY, newendX, newendY,\
        padding_left, padding_top, padding_right, padding_bottom)


def extract_face(img_path, out_face_img_path=None, out_face_size=256, initial_bb=None, suppress_stdout=True):
    img = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),\
        1.0, (300, 300), (104.0, 177.0, 123.0))

    MODEL.setInput(blob)
    detections = MODEL.forward()

    if detections.shape[2] > 0:
        try:
            if initial_bb is None:
                # print(1)
                # ---------- OBSOLETE ----------
                # Get the most confident bounding box
                i = 0   # extract the most confident face
                # confidence = detections[0, 0, i, 2]
                # print(confidence)
                (h, w) = img.shape[:2]
                bb = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # result_bb = bb
                # NOTE: Instead of getting the most confident bounding box,
                # we get the bounding box that has the highest iou with the
                # initial bounding box, which is guaranteed to have the face
                # ------------------------------
            else:
                # print(2)
                # Get the highest iou bounding box
                bb = get_bb(initial_bb, detections, *img.shape[:2], suppress_stdout=suppress_stdout)
            if not suppress_stdout:
                print('Individual bb: ', bb)

            # result_bb = bb.astype('int') if initial_bb is None\
            #     else get_min_union_bb(initial_bb, bb.astype('int'))
            result_bb = bb.astype('int') # don't union
            if not suppress_stdout:
                print('Union bb: ', result_bb)

            (startX, startY, endX, endY,\
                padding_left, padding_top, padding_right, padding_bottom)\
                    = make_square_img(*bb.astype('int'), *img.shape[:2], min_size=out_face_size)
            if not suppress_stdout:
                print('Square bb: ', startX, startY, endX, endY,\
                    padding_left, padding_top, padding_right, padding_bottom)

            # result_bb = (startX, startY, endX, endY) if initial_bb is None\
            #     else get_min_union_bb(initial_bb, (startX, startY, endX, endY))
            # result_bb = (startX, startY, endX, endY) # don't union
            # print('Union bb: ', result_bb)

            # pad image
            if max(padding_left, padding_top, padding_right, padding_bottom) > 0:
                img = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right,\
                    cv2.BORDER_REPLICATE, None, None)

            # crop image
            # (startX, startY, endX, endY) = bb.astype('int')
            face_img = img[startY:endY, startX:endX]

            # resize to desired size
            try:
                # print('out_face_size:',out_face_size)
                face_img = cv2.resize(face_img, (out_face_size, out_face_size))
            except Exception as e:
                print(e)
            # print(face_img.shape)

            # save or show image
            if out_face_img_path is not None:
                cv2.imwrite(out_face_img_path, face_img)
            else:
                cv2.imshow("Output", face_img)
                cv2.waitKey(0)

            return 1, result_bb
        except:
            return 0, ()
    else:
        return 0, ()


def extract_faces_in_dir(dir_path, out_dir_path=None, out_face_size=256, initial_bb=None, suppress_stdout=True):
    # check input params
    if not os.path.isdir(dir_path):
        return 0
    if out_dir_path == None:
        dirname = os.path.basename(dir_path)
        out_dir_path = get_path_same_dir(dir_path, dirname + '_faces')
    ensure_dir(out_dir_path)
    print('Processing \"{}\"...'.format(dir_path))

    # do work
    count = 0
    union_bb = initial_bb
    for img_path in tqdm(sorted(os.listdir(dir_path))):
        img_name, img_ext = os.path.splitext(img_path)
        if img_ext in ('.png', '.jpg'):
            img_abs_path = os.path.join(dir_path, img_path)
            if not suppress_stdout:
                print('=============================================================')
                print(img_abs_path)
            out_img_path = os.path.join(out_dir_path, img_path)
            if not suppress_stdout:
                print('Old union bb:', union_bb)
            success, bb = extract_face(img_abs_path, out_face_img_path=out_img_path, out_face_size=out_face_size, initial_bb=union_bb, suppress_stdout=suppress_stdout)
            union_bb = bb
            if not suppress_stdout:
                print('New union bb:', union_bb)
                print('=============================================================')
            count += success
    print("{} faces saved to \"{}\"".format(count, out_dir_path))

    return count


def get_time_step(img_path, initial_bb=None, suppress_stdout=True):
    img = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),\
        1.0, (300, 300), (104.0, 177.0, 123.0))

    MODEL.setInput(blob)
    detections = MODEL.forward()

    result_bb = (-1, -1, -1, -1)
    if detections.shape[2] > 0:
        try:
            if initial_bb is None:
                # print(1)
                # ---------- OBSOLETE ----------
                # Get the most confident bounding box
                i = 0   # extract the most confident face
                # confidence = detections[0, 0, i, 2]
                # print(confidence)
                (h, w) = img.shape[:2]
                bb = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                # result_bb = bb
                # NOTE: Instead of getting the most confident bounding box,
                # we get the bounding box that has the highest iou with the
                # initial bounding box, which is guaranteed to have the face
                # ------------------------------
            else:
                # print(2)
                # Get the highest iou bounding box
                bb = get_bb(initial_bb, detections, *img.shape[:2], suppress_stdout=suppress_stdout)

            result_bb = bb.astype('int') # don't union
            return result_bb
        except:
            return result_bb
    else:
        return result_bb


def get_time_series(dir_path, initial_bb=None, suppress_stdout=True):
    # check input params
    if not os.path.isdir(dir_path):
        return 0
    print('Processing \"{}\"...'.format(dir_path))

    # do work
    img_paths = []
    startXs = []
    startYs = []
    endXs = []
    endYs = []
    union_bb = initial_bb
    for img_path in tqdm(sorted(os.listdir(dir_path))):
        img_name, img_ext = os.path.splitext(img_path)
        if img_ext in ('.png', '.jpg'):
            img_abs_path = os.path.join(dir_path, img_path)
            img_paths.append(img_abs_path)
            if not suppress_stdout:
                print('=============================================================')
                print(img_abs_path)
            if not suppress_stdout:
                print('Old union bb:', union_bb)
            bb = get_time_step(img_abs_path, initial_bb=union_bb, suppress_stdout=suppress_stdout)
            startXs.append(bb[0])
            startYs.append(bb[1])
            endXs.append(bb[2])
            endYs.append(bb[3])
            union_bb = bb
            if not suppress_stdout:
                print('New union bb:', union_bb)
                print('=============================================================')

    return img_paths, startXs, startYs, endXs, endYs


def extract_faces_by_bb(img_paths, startXs, startYs, endXs, endYs, out_face_size=256, overwrite=False, suppress_stdout=True):
    assert len(img_paths) == len(startXs) == len(startYs) == len(endXs) == len(endYs) > 0

    # smooth time series
    sigma = 3
    startXs = list(ndimage.filters.gaussian_filter1d(startXs, sigma))
    startYs = list(ndimage.filters.gaussian_filter1d(startYs, sigma))
    endXs = list(ndimage.filters.gaussian_filter1d(endXs, sigma))
    endYs = list(ndimage.filters.gaussian_filter1d(endYs, sigma))

    count = 0
    out_dir_path = os.path.join(get_parent_dir(get_parent_dir(img_paths[0])),\
        os.path.basename(get_parent_dir(img_paths[0])) + '_faces')
    if not os.path.exists(out_dir_path) or overwrite or len(img_paths) != get_number_of_files(out_dir_path):
        ensure_dir(out_dir_path)

        try:
            for i, bb in enumerate(tqdm(list(zip(startXs, startYs, endXs, endYs)))):
                img = cv2.imread(img_paths[i])
                out_face_img_path = os.path.join(out_dir_path, os.path.basename(img_paths[i]))
                startX, startY, endX, endY = bb

                # fill surrounding
                (startX, startY, endX, endY,\
                    padding_left, padding_top, padding_right, padding_bottom)\
                        = make_square_img(*bb, *img.shape[:2], min_size=out_face_size)

                # pad image
                if max(padding_left, padding_top, padding_right, padding_bottom) > 0:
                    img = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right,\
                        cv2.BORDER_REPLICATE, None, None)

                # crop image
                face_img = img[startY:endY, startX:endX]

                # resize to desired size
                try:
                    face_img = cv2.resize(face_img, (out_face_size, out_face_size))
                except Exception as e:
                    print(e)
                # print(face_img.shape)

                # save or show image
                if out_face_img_path is not None:
                    cv2.imwrite(out_face_img_path, face_img)
                else:
                    cv2.imshow("Output", face_img)
                    cv2.waitKey(0)
                
                count += 1
        except:
            print("{} faces saved to \"{}\"".format(count, out_dir_path))
            return count, out_dir_path

        print("{} faces saved to \"{}\"".format(count, out_dir_path))
        return count, out_dir_path

    else:
        count = get_number_of_files(out_dir_path)
        print("{} faces saved to \"{}\"".format(count, out_dir_path))
        return count, out_dir_path


def extract_all_faces(dir_path, initial_bb=None, out_face_size=256, overwrite=False, suppress_stdout=True):
    """all-in-one function to call"""
    out_dir_path = os.path.join(get_parent_dir(dir_path), os.path.basename(dir_path) + '_faces')
    if not os.path.exists(out_dir_path) or overwrite\
        or get_number_of_files(dir_path) != get_number_of_files(out_dir_path):
        return extract_faces_by_bb(\
            *get_time_series(dir_path, initial_bb=initial_bb),\
                out_face_size=out_face_size, overwrite=overwrite, suppress_stdout=suppress_stdout)
    else:
        return get_number_of_files(out_dir_path), out_dir_path


def process_json_data(input_json_path, out_face_size=256, output_json_path=None, overwrite=False):
    print('Processing \"{}\"...'.format(input_json_path))
    dataset_path = ''
    num_videos = 0
    all_info = []
    with open(input_json_path, 'r') as fp:
        json_obj = json.load(fp)
        dataset_path = json_obj['dataset_path']
        num_videos = json_obj['num_videos']
        all_info = json_obj['files']

    # print(dataset_path)
    # print(num_videos)
    # print(len(all_info))
    # print(all_info[0])

    output_json_obj = OrderedDict()
    output_json_obj['dataset_path'] = dataset_path
    output_json_obj['num_videos'] = num_videos
    # output_json_obj['files'] = []
    output_json_clips = []
    for clip in tqdm(all_info):
        
        # print(clip['face_x'])
        # print(clip['face_y'])
        # print(clip['frames_path'])
        face_x = float(clip['face_x'])
        face_y = float(clip['face_y'])
        cur_dir = clip['frames_path']

        # print(os.path.join(cur_dir, sorted(os.listdir(cur_dir))[0]))
        initial_bb = get_initial_bb(os.path.join(cur_dir, sorted(os.listdir(cur_dir))[0]), face_x, face_y)
        print('Initial bb:', initial_bb)
        # success, out_dir = extract_faces_by_bb(*get_time_series(cur_dir, initial_bb=initial_bb), out_face_size=out_face_size)
        success, out_dir = extract_all_faces(cur_dir, initial_bb=initial_bb, out_face_size=out_face_size, overwrite=overwrite)

        clip['frames_path'] = out_dir
        output_json_clips.append(clip)
    output_json_obj['files'] = output_json_clips

    # write file
    if output_json_path is None:
        output_json_path = get_path_same_dir(input_json_path, os.path.basename(input_json_path).split('.json')[0] + '_faces.json')
    print('Writing to "{}"...'.format(output_json_path))
    with open(output_json_path, 'wb') as f:
        json_str = json.dumps(output_json_obj, **JSON_DUMP_PARAMS)
        f.write(json_str.encode())

    print('Done')


if __name__ == "__main__":
    # extract_face('/home/henry/Desktop/1/0000130.jpg')

    # face_x = 0.270313
    # face_y = 0.219444
    # initial_bb = get_initial_bb('/home/henry/Desktop/1/0000001.jpg', face_x, face_y)
    # print('Initial bb:', initial_bb)
    # # extract_faces_in_dir('/home/henry/Desktop/1', initial_bb=initial_bb, suppress_stdout=False)
    # extract_faces_by_bb(*get_time_series('/home/henry/Desktop/1', initial_bb=initial_bb), out_face_size=256)

    # face_x = 0.8265620000000001
    # face_y = 0.191667
    # initial_bb = get_initial_bb('/home/henry/Desktop/2/0000001.jpg', face_x, face_y)
    # print('Initial bb:', initial_bb)
    # # extract_faces_in_dir('/home/henry/Desktop/2', out_face_size=256, initial_bb=initial_bb)
    # extract_faces_by_bb(*get_time_series('/home/henry/Desktop/2', initial_bb=initial_bb), out_face_size=256)

    # face_x = 0.770312
    # face_y = 0.34444400000000003
    # initial_bb = get_initial_bb('/home/henry/Desktop/3/0000001.jpg', face_x, face_y)
    # print('Initial bb:', initial_bb)
    # # extract_faces_in_dir('/home/henry/Desktop/3', initial_bb=initial_bb, suppress_stdout=True)
    # extract_faces_by_bb(*get_time_series('/home/henry/Desktop/3', initial_bb=initial_bb), out_face_size=256)

    TED_TRAINING_JSON = os.path.join(DATA_ROOT, 'training_TED.json')
    TED_TESTING_JSON = os.path.join(DATA_ROOT, 'testing_TED.json')

    # process_json_data(TED_TESTING_JSON)
    process_json_data(TED_TRAINING_JSON)
