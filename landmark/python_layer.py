import caffe
import numpy as np
import copy
import random
import config
import sys
from batch_loader import BatchLoader

class LandmarkDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        self.batch = int(param['batch'])
        self.img_size = config.IMG_SIZE

        self.batch_loader = BatchLoader(param, "train")
        self.ohem_batch_loader = BatchLoader(param, 'ohem')
        self.train_ratio = 1.
        self.ohem_ratio = 0
        if self.ohem_batch_loader.is_loaded():
            self.train_ratio = 7/8.
            self.ohem_ratio = 1. - self.train_ratio
        top[0].reshape(self.batch, 3, self.img_size, self.img_size)  # data
        top[1].reshape(self.batch, config.LANDMARK_SIZE * 2)  # landmark
        top[2].reshape(self.batch, 1)  # landmark

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        batch_data = self.batch_loader.next_batch(self.batch * self.train_ratio, '')
        batch_data += self.ohem_batch_loader.next_batch(self.batch * self.ohem_ratio, '')
        for i, datum in enumerate(batch_data):
            img, pts, eye_dist = datum
            top[0].data[i, ...] = img
            top[1].data[i, ...] = pts
            top[2].data[i, ...] = eye_dist

    def backward(self, bottom, top):
        pass


class LandmarkLossLayer(caffe.Layer):
    def setup(self,bottom,top):
        self.top_ratio = config.TOP_DIFF_RATIO
        self.inner_feature_ratio = config.INNER_FEATURE_RATIO
        if len(bottom) < 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension %d %d" % (bottom[0].count, bottom[1].count))

        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.selected_diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # self.eye_dist = np.zeros_like(bottom[2].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = 0
        top[0].data[...] = 0
        # self.eye_dist[...] = bottom[2].data
        self.diff[...] = (bottom[0].data - bottom[1].data) #/ bottom[2].data
        self.diff[:, 24:] *= self.inner_feature_ratio
        sqr_diff = self.diff ** 2
        top_k = int(self.top_ratio * len(self.diff))
        if self.top_ratio > 0:
            l2_diff = np.sum(sqr_diff, axis=-1)
            indices = np.argpartition(l2_diff, -top_k)[-top_k:]
            self.selected_diff[indices] = self.diff[indices]
            sqr_diff = self.selected_diff ** 2

        top[0].data[...] = np.sum(sqr_diff) / bottom[0].num   / 2.

    def backward(self,top,propagate_down,bottom):
        for i, pd in enumerate(propagate_down):
            if pd == 0:
                continue
            sign = 1 if i == 0 else -1
            bottom[i].diff[...] = sign * self.diff  / bottom[i].num

