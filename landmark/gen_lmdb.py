import sys
import os
import lmdb
import cv2
import numpy as np
import landmark_pb2
import config
from scipy.spatial import distance


def write_lmdb(txn, img, pts, index):

    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = np.swapaxes(img, 0, 2)
    img = (img - 127.5) / 127.5
    datum = landmark_pb2.Datum()
    datum.img = img.tobytes()
    for pt in pts:
        x, y = tuple(pt)
        datum.pts.append(x / float(w))
        datum.pts.append(y / float(h))
    datum.eye_dist = distance.euclidean((datum.pts[38*2], datum.pts[38*2+1]),
                                        (datum.pts[21*2], datum.pts[21*2+1]))

    txn.put(str(index).encode('ascii'), datum.SerializeToString())

if __name__ == '__main__':
    env = lmdb.open(config.DB_DIR % 'train', map_size=config.DB_MAPSIZE)
    anno_file = os.path.join(config.DATA_DIR % 'train', 'landmarks.txt')
    with env.begin(write=True) as txn:
        with open(anno_file, 'r') as f:
            for i, line in enumerate(f):
                segs = line.strip().split()
                if len(segs) != config.LANDMARK_SIZE * 2 + 1:
                    print("invalid line %d length %d" % (i, len(segs)))
                    continue
                img_path = segs[0]
                img = cv2.imread(img_path)
                h, w, c = img.shape
                pts = np.asarray(map(float, segs[1:])).reshape((config.LANDMARK_SIZE, 2))
                try:
                    write_lmdb(txn, img, pts, i)
                except:
                    print("invalid img %s %d" % (img_path, i))
                if i % 100 == 0:
                    print('image %d done' % i)

