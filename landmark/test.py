import os
import sys
import cv2
import time
import caffe
import numpy as np
import config
sys.path.append('../')
from fast_mtcnn import fast_mtcnn
from gen_landmark import expand_mtcnn_box, is_valid_facebox, extract_baidu_lm72
from baidu import call_baidu_api

def create_net(model_dir, iter_num):
    model_path = os.path.join(model_dir, 'landmark_iter_%d.caffemodel' % iter_num)
    proto_path = 'landmark.prototxt'
    return caffe.Net(proto_path, model_path, caffe.TEST)

if __name__ == '__main__':
    iter_num = int(sys.argv[1])
    img_path = sys.argv[2]
    model_dir = config.MODEL_DIR
    if len(sys.argv) > 3:
        model_dir = sys.argv[3]

    img = cv2.imread(img_path)
    net = create_net(model_dir, iter_num)

    mtcnn = fast_mtcnn()
    boxes = mtcnn(img_path)
    for box in boxes:
        if not is_valid_facebox(box):
            continue
        exp_box = expand_mtcnn_box(img, box)
        cropped = img[exp_box[1]:exp_box[3], exp_box[0]:exp_box[2]]
        baidu_result = call_baidu_api(cropped, '')
        baidu_lm = extract_baidu_lm72(baidu_result[0][-1])
        for x, y in baidu_lm:
            x = int(x + exp_box[0])
            y = int(y + exp_box[1])
            cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 1)

        h, w, _ = cropped.shape
        cropped = cv2.resize(cropped, (config.IMG_SIZE, config.IMG_SIZE))
        cropped = np.swapaxes(cropped, 0, 2)
        cropped = (cropped - 127.5) / 127.5
        net.blobs['data'].data[0] = cropped
        out = net.forward()
        landmark = out['Dense2'][0]
        for pt in landmark.reshape((config.LANDMARK_SIZE, 2)):
            x, y = pt
            x = x * w + exp_box[0]
            y = y * h + exp_box[1]
            cv2.circle(img, (int(x), int(y)), 1, (255, 255, 0), 1)
        time.sleep(0.5)

    cv2.imwrite('result.jpg', img)

