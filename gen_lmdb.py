import os
import sys
import cv2
import random
import numpy as np
import lmdb

import config as config
import mtcnn_pb2

def write_lmdb(txn, lines, net, dt):
    net_size = config.NET_IMG_SIZES[net]
    img_size = net_size, net_size
    count = 0
    total = 0
    for num, line in enumerate(lines):
        total += 1
        segs = line.split()
        img_path = segs[0]
        img = cv2.imread(img_path)
        h, w, c = img.shape
        if (h, w) != img_size:
            img = cv2.resize(img, img_size)
        img = np.swapaxes(img, 0, 2)
        img = (img - 127.5) / 127.5
        label = int(segs[1])
        bbox  = [-1.0] * 4
        landm5 = [-1.0] * config.LANDMARK_SIZE * 2
        datum = mtcnn_pb2.Datum()
        datum.img = img.tobytes()
        if label != config.DATA_TYPES['neg']:
            if len(segs) >= 6:
                bbox = [float(x) for x in segs[2:6]]
            else:
                print('no enough bbox data in line %d' % num)
                continue
        if label == config.DATA_TYPES['landm5']:
            if len(segs) == 6 + config.LANDMARK_SIZE * 2:
                pts = [float(x) for x in segs[6:]]
            else:
                print('no enough landm5ark data in line %d' % num)
                continue
        datum.label = label
        datum.bbox[:] = bbox
        datum.landm5[:] = landm5
        datum.c, (datum.w, datum.h) = c, img_size

        txn.put(str(count).encode('ascii'), datum.SerializeToString())
        count += 1
        if count % 1000 == 0:
            print("%d imgs done" % count)

    print("write lmdb for %s %s, total %d write %d" % (net, dt, total, count))


if __name__ == '__main__':
    net = sys.argv[1]
    data_dir = config.TRAIN_DATA_DIR
    dts = config.DATA_TYPES.keys()
    try:
        os.makedirs(config.DB_PATH)
        os.makedirs('%s/%s' % (config.DB_PATH, net))
    except:
        pass

    for dt in dts:
        env = lmdb.open('%s/%s/%s.lmdb' % (config.DB_PATH, net, dt), map_size=config.DB_MAPSIZES[dt])
        with env.begin(write=True) as txn:
            img_list = '%s/%s/%s_%s.txt' % (data_dir, net, dt, net)
            print("open imglist ", img_list)
            with open(img_list, 'r') as f:
                write_lmdb(txn, f.readlines(), net, dt)

