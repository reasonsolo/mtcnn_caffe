import caffe
import numpy as np
import copy
import random
import config
from batch_loader import BatchLoader

class LandmarkDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        self.batch = int(param['batch'])

        self.batch_loader = BatchLoader(param, "train")
        self.ohem_batch_loader = BatchLoader(param, 'ohem')
        self.train_ratio = 1.
        self.ohem_ratio = 0
        if self.ohem_batch_loader.is_loaded():
            self.train_ratio = 7/8.
            self.ohem_ratio = 1. - self.train_ratio
        top[0].reshape(self.batch, 3, self.img_size, self.img_size)  # data
        top[1].reshape(self.batch, 72)  # landmark

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        batch_data = self.batch_loader.next_batch(self.batch * self.train_ratio, '')
        batch_data += self.ohem_batch_loader.next_batch(self.batch * self.ohem_ratio, '')
        random.shuffle(batch_data)
        for i, datum in enumerate(batch_data):
            img, label, bbox, landm5 = datum
            top[0].data[i, ...] = img
            top[1].data[i, ...] = label
            top[2].data[i, ...] = bbox
            if self.net != 'pnet':
                top[3].data[i, ...] = landm5

    def backward(self, bottom, top):
        pass


class LandmarkLossLayer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")

        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = 0
        top[0].data[...] = 0
        self.eye_dist[...] = np.linalg.norm(bottom[1].data[:,[38]] - bottom[1].data[:,[21]], axis=2)
        self.diff[...] = (bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)) / self.eye_dist[:, None]
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self,top,propagate_down,bottom):
        for i, pd in enumerate(propagate_down):
            if pd == 0:
                continue
            sign = 1 if i == 0 else -1
            bottom[i].diff[...] = sign * self.diff  / bottom[i].num

