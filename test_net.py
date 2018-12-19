import cv2
import caffe
import tools
import sys
import os
import numpy as np
from functools import partial

import config

def gen_scales(w, h, min_imgsize, net_imgsize):
    scales = []
    scale = float(net_imgsize) / min_imgsize;
    minhw = min(w, h) * scale;

    while minhw > net_imgsize:
        scales.append(scale)
        scale *= 0.709
        minhw *= 0.709
    return scales

def regularize_rect(img_shape, rect):
    w, h, _ = img_shape
    x1, y1, x2, y2, prob = rect
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    return [x1, x2, y1, y2, prob]


def load_model(model_dir, net, iter_num):
    proto_path = os.path.join(model_dir, '%s.prototxt' % net)
    model_path = os.path.join(model_dir, '%s_iter_%d.caffemodel' % (net, iter_num))
    return caffe.Net(proto_path, model_path, caffe.TEST)


def test_pnet(img, min_img_size, net_size, net):
    norm_img = (img.copy() - 127.5) / 128
    h, w, c = norm_img.shape
    scales = gen_scales(w, h, min_img_size, net_size)
    rects = []
    for scale in scales:
        sh = int(h * scale)
        sw = int(w * scale)
        scale_img = cv2.resize(norm_img, (sw, sh))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net.blobs['data'].reshape(1, 3, sw, sh)
        net.blobs['data'].data[...] = scale_img
        out = net.forward()
        label_prob = out[config.NET_OUTPUTS['pnet']['label']][0][1]
        bbox = out[config.NET_OUTPUTS['pnet']['bbox']][0]
        out_h, out_w = label_prob.shape
        out_side = max(out_h, out_w)
        rect = tools.detect_face_12net(label_prob, bbox, out_side,
                                       1 / scale, w, h, 0.7)
        rects += rect

    rects = tools.NMS(rects, 0.7, 'iou')
    return rects

def test_rnet(img, rects, min_img_size, net_size, net):
    norm_img = (img.copy() - 127.5) / 128
    h, w, c = norm_img.shape
    for i, rect in enumerate(rects):
        cropped_img = img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2]),]
        resized_img = cv2.resize(cropped_img, (net_size, net_size))
        resized_img = np.swapaxes(resized_img, 0, 2)
        net.blobs['data'].data[i] = resized_img
    out = net.forward()
    label_prob = out[config.NET_OUTPUTS['rnet']['label']][0][1]
    bbox = out[config.NET_OUTPUTS['rnet']['bbox']][0][1]
    rects = tools.filter_face_24net(label_prob, bbox, rects, w, h, 0.7)

    return [rect for rect in rects if rects[2] - rects[0] > 0 and rects[3] - rects[1] > 0]


def test_net(net, model_dir, iter_num):
    model_path = os.path.join(model_dir, '%s_iter_%d.caffemodel' % (net, iter_num))
    proto_path = os.path.join(model_dir, '%s.prototxt' % net)
    caffe_net = caffe.Net(proto_path, model_path, caffe.TEST)
    if net == 'pnet':
        return partial(test_pnet, net=caffe_net)
    elif net == 'rnet':
        return partial(test_rnet, net=caffe_net)

if __name__ == '__main__':
    net = sys.argv[1]
    iter_num = int(sys.argv[2])

    test_func = test_net(net, config.MODEL_DIR, iter_num)
    img_path = sys.argv[3]
    img = cv2.imread(img_path)

    rects = test_func(img, config.MIN_IMG_SIZE, config.NET_IMG_SIZES['pnet'])
    for i, rect in enumerate(rects):
        sub_img = img[rect[1]:rect[3], rect[0]:rect[2]]
        print(sub_img.shape, rect)
        cv2.imwrite("pnet/test/%d_%f.jpg" % (i, rect[4]), sub_img)
    print('%d rects generated' % len(rects))

