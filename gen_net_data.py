import sys
import os
import numpy as np
import numpy.random as npr
import cv2

import config
from dataset_loader import iterate_wider, iterate_lfw
from utils import IoU


def usage():
    print('%s [pnet|rnet|onet] [train|val|test]' % sys.argv[0])

def wider_iter_func(wider_dir, save_dir, img_size, indices, files):
    def gen_data(img_name, bboxes):
        img_path = os.path.join(wider_dir, 'images', img_name)
        img = cv2.imread(img_path)
        nboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

        # cropped box and bounding box
        for (cbox, size), bbox in gen_crop_boxes(img, nboxes, img_size):
            if cbox is None:
                continue
            iou = IoU(cbox, nboxes)
            store_gen_box(save_dir, img, cbox, size, bbox, iou, files, indices)

        print("generate %s subimages for %s" % (str(indices), img_path))
    return gen_data

def lfw_iter_func(lfw_dir, save_dir, img_size, indices, files, with_landm5=True):
    print("gen data for lfw with landmark5 %s" % with_landm5)
    def gen_data(img_name, bboxes, landm5):
        img_path = os.path.join(lfw_dir, img_name)
        print(img_path)
        img = cv2.imread(img_path)
        nboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        nlandm5s = np.array(landm5, dtype=np.float32).reshape(-1, 5, 2)

        # cropped box and bounding box
        # for (cbox, size), bbox in gen_crop_boxes(img, nboxes, img_size):
        #     if cbox is None:
        #         continue
        #     iou = IoU(cbox, nboxes)
        #     store_gen_box(save_dir, img, cbox, size, bbox, iou, files, indices)
        for (cbox, size), bbox, clandm5 in gen_crop_landm5(img, nboxes, nlandm5s, img_size):
            if cbox is None:
                continue
            iou = IoU(cbox, nboxes)
            store_gen_box_lfw(save_dir, img, cbox, size, bbox, iou, clandm5, files, indices, with_landm5)

        print("generate %s subimages for %s" % (str(indices), img_path))
    return gen_data

def gen_crop_landm5(img, bboxes, landm5s, img_sizes):
    for bbox, landm5 in zip(bboxes, landm5s):
        for i in range(0, 5):
            cbox, size = gen_pos_box(img, img_size, bbox)
            offset_x, offset_y = cbox[0] - bbox[0], cbox[1] - bbox[1]
            transformed_landm5 = [(x - offset_x -bbox[0], y - offset_y - bbox[1]) for x, y in landm5]
            yield (cbox, size), bbox, transformed_landm5

def gen_crop_boxes(img, bboxes, img_size):
    for i in range(0, 50):
        yield gen_rand_box(img, img_size), None
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # omit invalid box or too small box
        if max(w, h) < 40 or min(w, h) / 2 <= img_size or x1 < 0 or y1 <0:
            continue
        for i in range(0, 4):
            yield gen_neg_box(img, img_size, bbox), bbox
        for i in range(0, 16):
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
    if img_h == 0 or  img_w ==  0:
        raise Exception("invalid shape %s" % str(img.shape))

    try:
        size = npr.randint(min(box_w, box_h) * 0.8, max(box_w, box_h) * 1.25)
    except Exception as ex:
        print(min(box_w, box_h) * 0.8, max(box_w, box_h) * 1.25, box)
        raise ex
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
    dt, cropped_img, label = gen_img_label(img, cbox, size, bbox, iou)
    if dt != '':
        resized_img = cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_LINEAR)
        save_img_file = os.path.join(save_dir, dt, '%s.jpg' % indices[dt])
        cv2.imwrite(save_img_file, resized_img)
        data_files[dt].write(save_img_file + " " + label)
        indices[dt] += 1
    return dt

def gen_img_label(img, cbox, size, bbox, iou, landm5=None, with_landm5=False):
    dt = ''
    label = ''
    max_iou = np.max(iou)
    norm_landm5 = None
    x1, y1, x2, y2 = cbox
    cropped_img = img[y1:y2, x1:x2]
    if bbox is None or max_iou < 0.3:
        dt = 'neg'
        label = "%d\n" % (config.DATA_TYPES[dt])
    elif max_iou > 0.65:
        if with_landm5 and len(landm5) == 5:
            dt = 'landm5'
            left = cbox[0]
            top = cbox[1]
            offset = "%.6f %.6f %.6f %.6f" % tuple([float(x - nx) / size for x, nx in zip(bbox, cbox)])
            norm_landm5 = [(float(x) / size, float(y) / size) for x, y in landm5]
            # flip over x axis
            if npr.choice([0, 1]) == 1:
                cropped_img = cv2.flip(cropped_img.copy(), 1)
                flipped_landm5 = np.asarray([(1-x, y) for x, y in norm_landm5])
                flipped_landm5[[0, 1]] = flipped_landm5[[1, 0]]
                flipped_landm5[[3, 4]] = flipped_landm5[[4, 3]]
                norm_landm5 = flipped_landm5.reshape(5, 2)
            landm5_label = ' '.join(['%0.6f' % x for point in norm_landm5 for x in point])
            label = "%d %s %s\n" % (config.DATA_TYPES[dt], offset, landm5_label)
        else:
            dt = 'pos'
            offset = "%.6f %.6f %.6f %.6f" % tuple([float(x - nx) / size for x, nx in zip(bbox, cbox)])
            label = "%d %s\n" % (config.DATA_TYPES[dt], offset)
    elif max_iou > 0.4:
        dt = 'part'
        offset = "%.6f %.6f %.6f %.6f" % tuple([float(x - nx) / size for x, nx in zip(bbox, cbox)])
        label = "%d %s\n" % (config.DATA_TYPES[dt], offset)

    return dt, cropped_img, label

def store_gen_box_lfw(save_dir, img, cbox, size, bbox, iou, landm5, data_files,
                      indices, with_landm5, store_img=False):
    dt, cropped_img, label = gen_img_label(img, cbox, size, bbox, iou, landm5, with_landm5)
    if dt != '':
        resized_img = cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_LINEAR)
        save_img_file = os.path.join(save_dir, dt, '%s.jpg' % indices[dt])
        cv2.imwrite(save_img_file, resized_img)
        data_files[dt].write(save_img_file + " " + label)
        indices[dt] += 1

        if store_img and dt == 'landm5':
            recovered_landm5 = [(int(x * size), int(y * size)) for (x, y) in norm_landm5]
            for pt in recovered_landm5:
                cv2.circle(cropped_img, pt, 2, (0,0,255), -1)
                save_img_file = os.path.join('landm5', '%s.jpg' % indices[dt])
            cv2.imwrite(save_img_file, cropped_img)
    return dt

if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
        sys.exit(-1)

    net = sys.argv[1]
    script_type = os.path.join(sys.argv[2])
    wider_dir = os.path.join(config.WIDER_DIR, 'WIDER_%s' % script_type)
    wider_anno_file = 'wider_face_%s_bbx_gt.txt' % script_type
    wider_anno_path = os.path.join(wider_dir, 'wider_face_split', wider_anno_file)
    lfw_dir   = os.path.join(config.LFW_DIR, 'lfw_%s' % script_type)
    lfw_anno_file = '%sImageList.txt' % script_type
    lfw_anno_path = os.path.join(lfw_dir, lfw_anno_file)
    save_dir  = os.path.join('data_%s' % script_type, net)
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
    indices = {}
    for dt in config.DATA_TYPES:
        indices[dt] = 0
        # overwride daa file
        files[dt] = open(os.path.join(save_dir, '%s_%s.txt' % (dt, net)), 'w')

    iterate_wider(wider_anno_path,  wider_iter_func(wider_dir, save_dir, img_size, indices, files))
    iterate_lfw(lfw_anno_path, lfw_iter_func(lfw_dir, save_dir, img_size, indices, files, True))

