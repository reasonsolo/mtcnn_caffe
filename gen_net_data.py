import sys
import os
import numpy as np
import numpy.random as npr
import cv2

import config
from wider_loader import iterate_wider
from utils import IoU


def usage():
    print('%s net wider_dir save_dir' % sys.argv[0])

def wider_iter_func(wider_dir, save_dir, img_size):
    files = {}
    indices = {}
    for dt in ['pos', 'neg', 'part']:
        indices[dt] = 0
        files[dt] = open(os.path.join(save_dir, '%s_%s.txt' % (dt, net)), 'w')

    def gen_data(img_name, bboxes):
        img_path = os.path.join(wider_dir, 'images', img_name)
        img = cv2.imread(img_path)
        nboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

        # cropped box and bounding box
        for (cbox, size), bbox in gen_crop_boxes(img, nboxes):
            if cbox is None:
                continue
            iou = IoU(cbox, nboxes)
            store_gen_box(save_dir, img, cbox, size, bbox, iou, files, indices)

        print("generate %s subimages for %s" % (str(indices), img_path))

    return gen_data

def gen_crop_boxes(img, bboxes):
    for i in range(0, 50):
        yield gen_rand_box(img, img_size), None
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # omit invalid box or too small box
        if max(w, h) < 40 or min(w, h) / 2 <= img_size or x1 < 0 or y1 <0:
            continue
        for i in range(0, 5):
            yield gen_neg_box(img, img_size, bbox), bbox
        for i in range(0, 20):
            yield gen_pos_box(img, img_size, bbox), bbox

def gen_rand_box(img, img_size):
    h, w, c = img.shape
    size = npr.randint(img_size, min(w, h) / 2)
    nx, ny = npr.randint(0, w - size), npr.randint(0, h - size)
    return [nx, ny, nx + size, ny + size], size

def gen_neg_box(img, img_size, box):
    x1, y1, x2, y2 = box
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1

    img_h, img_w, _ = img.shape
    try:
        size = npr.randint(img_size, min(box_w, box_h) / 2 + 1)
    except:
        print(box_w, box_h, img_size)
        raise ""
    delta_x = npr.randint(max(-size, -x1), box_w)
    delta_y = npr.randint(max(-size, -y1), box_h)
    nx1 = int(max(0, x1 + delta_x))
    ny1 = int(max(0, y1 + delta_y))

    if nx1 + size > img_w or ny1 + size > img_h:
        return None, None

    return [nx1, ny1, nx1 + size, ny1 + size], size

def gen_pos_box(img, img_size, box):
    x1, y1, x2, y2 = box
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1

    img_h, img_w, _ = img.shape

    size = npr.randint(min(box_w, box_h) * 0.8, max(box_w, box_h) * 1.25)
    # delta is the offset of box center
    delta_x = int(npr.randint(-box_w, box_w) * 0.2)
    delta_y = int(npr.randint(-box_h, box_h) * 0.2)

    nx1 = int(max(x1 + box_w / 2 + delta_x - size / 2, 0))
    nx2 = nx1 + size
    ny1 = int(max(y1 + box_h / 2 + delta_y - size / 2, 0))
    ny2 = ny1 + size
    if nx2 > img_w or ny2 > img_h:
        return None, None
    return [nx1, ny1, nx2, ny2], size

def store_gen_box(save_dir, img, cbox, size, bbox, iou, data_files, indices):
    if bbox is None or np.max(iou) < 0.3:
        dt = 'neg'
        label = "%d\n" % (config.DATA_TYPES[dt])
    elif np.max(iou) > 0.65:
        dt = 'pos'
        offset = "%.6f %.6f %.6f %.6f" % tuple([float(x - nx) / size for x, nx in zip(bbox, cbox)])
        label = "%d %s\n" % (config.DATA_TYPES[dt], offset)
    elif np.max(iou) > 0.4:
        dt = 'part'
        offset = "%.6f %.6f %.6f %.6f" % tuple([float(x - nx) / size for x, nx in zip(bbox, cbox)])
        label = "%d %s\n" % (config.DATA_TYPES[dt], offset)
    else:
        return
    x1, y1, x2, y2 = cbox
    croped_img = img[y1:y2, x1:x2]
    resized_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_LINEAR)
    save_img_file = os.path.join(save_dir, dt, '%s.jpg' % indices[dt])
    cv2.imwrite(save_img_file, resized_img)
    data_files[dt].write(save_img_file + " " + label)
    indices[dt] += 1
    return dt


if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
        os.exit(-1)

    anno_file = 'wider_face_val_bbx_gt.txt'
    net = sys.argv[1]
    wider_dir = os.path.join(sys.argv[2])
    anno_path = os.path.join(sys.argv[2], anno_file)
    save_dir  = os.path.join(sys.argv[3], net)
    try:
        os.makedirs(save_dir)
    except:
        pass
    try:
        for dt in config.DATA_TYPES:
            print('mkdir %s' % os.path.join(save_dir, dt))
            os.makedirs(os.path.join(save_dir, dt))
    except:
        pass
    img_size = config.NET_IMG_SIZES[net]
    print("generate %s data size %d read wider from %s save file to %s" \
          % (net, img_size, wider_dir, save_dir))
    iterate_wider(anno_path,  wider_iter_func(wider_dir, save_dir, img_size))

