import cv2
import caffe
import tools
import sys
import numpy as np

def gen_scales(w, h, min_imgsize, net_imgsize):
    scales = []
    scale = float(net_imgsize) / min_imgsize;
    minhw = min(w, h) * scale;

    while minhw > net_imgsize:
        scales.append(scale)
        scale *= 0.709
        minhw *= 0.709
    return scales

def pnet(img, net):
    norm_img = (img.copy() - 127.5) / 128
    h, w, c = norm_img.shape
    scales = gen_scales(w, h, 40, 12)
    rects = []
    for scale in scales:
        print(scale)
        sh = int(h * scale)
        sw = int(w * scale)
        scale_img = cv2.resize(norm_img, (sw, sh))
        scale_img = np.swapaxes(scale_img, 0, 2)
        net.blobs['data'].reshape(1, 3, sw, sh)
        net.blobs['data'].data[...] = scale_img
        out = net.forward()
        label_prob = out['prob1'][0][1]
        bbox = out['conv4-2'][0]
        print(label_prob.shape)
        print(bbox.shape)
        out_h, out_w = label_prob.shape
        out_side = max(out_h, out_w)
        rect = tools.detect_face_12net(label_prob, bbox, out_side,
                                       1 / scale, w, h, 0.7)
        rects += rect

    rects = tools.NMS(rects, 0.7, 'iou')

    print("rects")
    for i, rect in enumerate(rects):
        sub_img = img[rect[2]:rect[3], rect[0]:rect[1]]
        cv2.imwrite("pnet/test/%d_%f.jpg" % (i, rect[4]), sub_img)


if __name__ == '__main__':
    proto = sys.argv[1]
    model = sys.argv[2]
    net = caffe.Net(proto, model, caffe.TEST)
    img_path = sys.argv[3]
    img = cv2.imread(img_path)
    print(img.shape)
    pnet(img, net)
