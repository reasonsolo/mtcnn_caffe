import cv2
import caffe
import numpy as np
import config
from mtcnn.loaders import BatchLoader

class DataLayer:
    def setup(self, bottom, top):
        param = eval(self.param_str)
        self.batch = param['batch']
        self.net = param['net']
        self.img_size = config[self.net]

        self.batch_loader = BatchLoader(param)
        top[0].reshape(self.batch, 3, self.img_size, self.img_size)  # data
        top[1].reshape(self.batch, 1)  # label
        top[2].reshape(self.batch, 4)  # bbox
        if self.net != 'pnet':
            top[3].data[i, ...].reshape(self.batch, config.LANDMARK_SIZE * 2)

    def forward(self, bottom, top):
        task = npr.randint(0, 2)
        batch_data = self.batch_loader.next_batch(self.batch, task)

        for i, datum in enumerate(batch_data):
            img, label, bbox, landmark = datum.img, datum.label, datum.bbox, dataum.landmark
            top[0].data[i, ...] = np.frombuffer(img).reshape((3, self.img_size, self.img_size))
            top[1].data[i, ...] = label
            top[2].data[i, ...] = bbox
            if self.net != 'pnet':
                top[3].data[i, ...] = landmark

    def backward(self, bottom, top):
        pass


class LabelBridgeLayer:
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self, bottom, top):
        label = bottom[1].data
        self.valid_index = np.where(label != config.DATA_TYPES['neg'])[0]
        self.count = len(self.valid_index)
        top[0].reshape(len(bottom[1].data), 2, 1, 1)
        top[1].reshape(len(bottom[1].data), 1)

    def forward(self, bottom, top):
        top[0].data[...][...]=0
        top[1].data[...][...]=0
        top[0].data[0:self.count] = bottom[0].data[self.valid_index]
        top[1].data[0:self.count] = bottom[1].data[self.valid_index]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0] and self.count!=0:
            bottom[0].diff[...]=0
            bottom[0].diff[self.valid_index]=top[0].diff[...]
        if propagate_down[1] and self.count!=0:
            bottom[1].diff[...]=0
            bottom[1].diff[self.valid_index]=top[1].diff[...]


class regression_Layer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")

        roi = bottom[1].data
        self.valid_index = np.where(roi[:,0] != -1)[0]
        self.N = len(self.valid_index)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(1)

    def forward(self,bottom,top):
        self.diff[...] = 0
        top[0].data[...] = 0
        if self.N != 0:
            self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
            top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self,top,propagate_down,bottom):
        for i in range(2):
            if not propagate_down[i] or self.N == 0:
                continue
            sign = 1 if i == 0 else -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


