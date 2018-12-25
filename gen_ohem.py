import sys
import os
import config
import cv2
import math
import random
import numpy as np
from collections import defaultdict

from utils import IoU
from gen_net_data import lfw_iter_func
from dataset_loader import iterate_wider, iterate_lfw
from test_net import test_net

def wider_hard_example_test(net, wider_dir, save_dir, detect_func, indices, files):
    prev_net = get_prev_net(net)
    test_img_size = config.NET_IMG_SIZES[prev_net]
    img_size = config.NET_IMG_SIZES[net]
    def gen_data(img_name, bboxes):
        img_path = os.path.join(wider_dir, 'images', img_name)
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        nboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        rects = detect_func(img, config.MIN_IMG_SIZE, test_img_size)
        img_dt_indices = defaultdict(int)
        random.shuffle(rects)
        for rect in rects:
            x1, y1, x2, y2, prob = rect
            w, h = x2 - x1 + 1, y2 - y1 +1
            if max(w, h) < config.MIN_IMG_SIZE / 2\
                or min(x1, y1, x2, y2) < 0\
                or max(x1, x2) >= img_w\
                or max(y1, y2) >= img_h\
                or x1 >= x2 or y1 >= y2:
                indices['filter_out'] += 1
                continue
            rect = convert_to_square(rect)
            iou = IoU(rect, nboxes)
            dt, label = gen_img_label(img, rect, bboxes, iou)
            if dt != '' and config.MAX_EXAMPLES[dt] > img_dt_indices[dt] :
                img_dt_indices[dt] += 1
                indices[dt] += 1
                cropped_img = img[rect[1]:rect[3], rect[0]:rect[2]]
                store_rect_data(save_dir, img_size, dt, cropped_img, label, files[dt], indices[dt])
            else:
                indices['ignore'] += 1
            if all(config.MAX_EXAMPLES[dt] == img_dt_indices[dt] for dt in config.DATA_TYPES):
                break
        print("generate %s subimages for %s rects %d" % (str(indices), img_path, len(rects)))
    return gen_data

def store_rect_data(save_dir, size, dt, cropped_img, label, f, index):
    resized_img = cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_LINEAR)
    save_img_file = os.path.join(save_dir, dt, '%s.jpg' % index)
    cv2.imwrite(save_img_file, resized_img)
    f.write(save_img_file + " " + label + "\n")

def convert_to_square(rect):
    center_x, center_y = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    w, h = rect[2] - rect[0] + 1, rect[3] - rect[1] + 1
    side = math.sqrt(w * h)
    square = [center_x - side / 2, center_y - side /2, center_x + side / 2, center_y + side / 2]
    return [int(x) for x in square]

def gen_img_label(img, rect, bboxes, iou):
    max_iou = np.max(iou)
    max_idx = np.argmax(iou)
    bbox = bboxes[max_idx]
    dt = ''
    label = ''
    if max_iou < 0.3:
        dt = 'neg'
        label = "%d" % (config.DATA_TYPES[dt])
    elif max_iou > 0.4:
        img_h, img_w, _ = img.shape
        crop_w, crop_h = rect[2] - rect[0], rect[3] - rect[1]
        zipped = list(zip(bbox, rect))
        offsets = [float(x1 - x2) / crop_w for x1, x2 in zipped[0:2]] +\
            [float(x1 - x2) / crop_h for x1, x2 in zipped[2:4]]
        offset = '%0.6f ' * 4 % tuple(offsets)
        if max_iou > 0.65:
            dt = 'pos'
        else:
            dt = 'part'
        label = '%d %s' % (config.DATA_TYPES[dt], offset)

    return dt, label

def get_prev_net(net):
    if net == 'pnet':
        prev_net = 'pnet'
    elif net == 'rnet':
        prev_net = 'pnet'
    elif net == 'pnet':
        prev_net = 'rnet'
    else:
        raise RuntimeError("no prev net for %s" % net)
    return prev_net

def usage():
    print("%s [rnet|pnet] [train|valid] iter_num" % sys.argv[0])
    exit(-1)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
    net = sys.argv[1]
    script_type = os.path.join(sys.argv[2])
    prev_net_iter_num = int(sys.argv[3])

    wider_dir = os.path.join(config.WIDER_DIR, 'WIDER_%s' % script_type)
    wider_anno_file = 'wider_face_%s_bbx_gt.txt' % script_type
    wider_anno_path = os.path.join(wider_dir, 'wider_face_split', wider_anno_file)
    lfw_dir   = os.path.join(config.LFW_DIR, 'lfw_%s' % script_type)
    lfw_anno_file = '%sImageList.txt' % script_type
    lfw_anno_path = os.path.join(lfw_dir, lfw_anno_file)
    save_dir  = os.path.join(config.DATA_DIR % 'ohem', net)
    try:
        os.makedirs(save_dir)
    except:
        pass
    for dt in config.DATA_TYPES:
        try:
            print('mkdir %s' % os.path.join(save_dir, dt))
            os.makedirs(os.path.join(save_dir, dt))
        except:
            pass
    img_size = config.NET_IMG_SIZES[net]
    print("generate %s data size %d read wider from %s save file to %s" \
          % (net, img_size, wider_dir, save_dir))
    # always iterate wider first
    files = {}
    indices = defaultdict(int)
    for dt in config.DATA_TYPES:
        # overwride daa file
        files[dt] = open(os.path.join(save_dir, '%s_%s.txt' % (dt, net)), 'w')

    test_func = test_net(get_prev_net(net), config.MODEL_DIR, prev_net_iter_num)
    iterate_wider(wider_anno_path,  wider_hard_example_test(net, wider_dir, save_dir, test_func, indices, files))
    # iterate_lfw(lfw_anno_path, lfw_iter_func(lfw_dir, save_dir, img_size, indices, files, with_landm5=True))

