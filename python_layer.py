import caffe
import numpy as np
import copy
import random
import config
from batch_loader import BatchLoader

class DataLayer(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        self.batch = int(param['batch'])
        self.net = param['net']
        self.img_size = config.NET_IMG_SIZES[self.net]

        self.batch_loader = BatchLoader(param)
        top[0].reshape(self.batch, 3, self.img_size, self.img_size)  # data
        top[1].reshape(self.batch, 1)  # label
        top[2].reshape(self.batch, 4)  # bbox
        top[3].reshape(self.batch, config.LANDMARK_SIZE * 2)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        task = random.choice(config.TRAIN_TASKS[self.net])
        batch_data = self.batch_loader.next_batch(self.batch, task)
        random.shuffle(batch_data)
        for i, datum in enumerate(batch_data):
            img, label, bbox, landm5 = datum
            top[0].data[i, ...] = img
            top[1].data[i, ...] = label
            top[2].data[i, ...] = bbox
            top[3].data[i, ...] = landm5

    def backward(self, bottom, top):
        pass


class LabelBridgeLayer(caffe.Layer):
    def setup(self, bottom, top):
        param = eval(self.param_str)
        self.net = param['net']

    def reshape(self, bottom, top):
        label = bottom[1].data
        self.label_valid_index = np.where(label!=config.DATA_TYPES['part'])[0]
        self.label_count = len(self.label_valid_index)
        top[0].reshape(len(bottom[0].data), 2, 1, 1)
        top[1].reshape(len(bottom[0].data), 1)
        self.landm5_index = np.where(label == config.DATA_TYPES['landm5'])[0]

        # self.pos_index = np.where(label == config.DATA_TYPES['pos'])[0]
        # self.part_index = np.where(label == config.DATA_TYPES['part'])[0]
        # self.neg_index = np.where(label == config.DATA_TYPES['neg'])[0]
        self.bbox_valid_index = np.where(label != config.DATA_TYPES['neg'])[0]
        self.bbox_count = len(self.bbox_valid_index)

        top[2].reshape(len(bottom[2].data), 4, 1, 1)
        top[3].reshape(len(bottom[2].data), 4)

        self.landm5_valid_index = np.where(label==config.DATA_TYPES['landm5'])[0]
        self.landm5_count = len(self.landm5_valid_index)
        top[4].reshape(len(bottom[4].data), config.LANDMARK_SIZE * 2, 1, 1)
        top[5].reshape(len(bottom[4].data), config.LANDMARK_SIZE * 2)

    def forward(self, bottom, top):
        # print(self.valid_index, self.part_index)
        # print(bottom[0].data[self.valid_index])
        self.label_count = len(self.label_valid_index)

        top[0].data[...][...] = 0
        top[1].data[...][...] = 1
        top[2].data[...][...] = -1
        top[3].data[...][...] = -1
        label_data = copy.copy(bottom[1].data)
        label_data[self.landm5_index] = 1    # regard all landm5 label as pos
        top[0].data[0:self.label_count] = bottom[0].data[self.label_valid_index]
        top[1].data[0:self.label_count] = label_data[self.label_valid_index]

        top[2].data[0:self.bbox_count] = bottom[2].data[self.bbox_valid_index]
        top[3].data[0:self.bbox_count] = bottom[3].data[self.bbox_valid_index]

        top[4].data[...][...] = -1
        top[5].data[...][...] = -1
        top[4].data[0:self.landm5_count] = bottom[4].data[self.landm5_valid_index]
        top[5].data[0:self.landm5_count] = bottom[5].data[self.landm5_valid_index]

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = 0
        bottom[1].diff[...] = 0
        for i, pd in enumerate(propagate_down[0:2]):
            if pd == 1 and self.label_count != 0:
                bottom[i].diff[self.label_valid_index] = top[i].diff[0:self.label_count]
                # for j in self.valid_index:
                #     bottom[i].diff[j] = top[i].diff[j]
        bottom[2].diff[...] = np.zeros(bottom[2].diff.shape)
        bottom[3].diff[...] = np.zeros(bottom[3].diff.shape)
        for i, pd in enumerate(propagate_down[2:4]):
            if pd == 1 and self.bbox_count != 0:
                i = i+2
                bottom[i].diff[self.bbox_valid_index] = top[i].diff[0:self.bbox_count]

        bottom[4].diff[...] = np.zeros(bottom[4].diff.shape)
        bottom[5].diff[...] = np.zeros(bottom[5].diff.shape)
        for i, pd in enumerate(propagate_down[4:6]):
            if pd == 1 and self.landm5_count != 0:
                i = i+4
                bottom[i].diff[self.landm5_valid_index] = top[i].diff[0:self.landm5_count]


class RegressionLossLayer(caffe.Layer):
    def setup(self,bottom,top):
        if len(bottom) != 2:
            raise Exception("Need 2 Inputs")

    def reshape(self,bottom,top):
        if bottom[0].count != bottom[1].count:
            raise Exception("Input predict and groundTruth should have same dimension")

        roi = bottom[1].data
        self.valid_index = np.where(roi[:,0] != config.DATA_TYPES['neg'])[0]
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
        for i, pd in enumerate(propagate_down):
            if pd == 0 or self.N == 0:
                continue
            sign = 1 if i == 0 else -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num


