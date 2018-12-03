import config
import os
import cv2
import lmdb
import random
import mtcnn_pb2
import numpy as np

class BatchLoader:
    def __init__(self, param):
        self.net = param['net']
        self.img_size = config.NET_IMG_SIZES[self.net]
        self.db_envs = {}
        self.db_path = os.path.join(config.DB_PATH, self.net)
        self.db_entries = {}

        self.label_ratio = {}
        self.db_indices = {}

        self.total_entries = 0
        self.total_labels = 0
        self.total_bboxes = 0
        self.total_landmark = 0
        for dt in config.DATA_TYPES:
            self.db_envs[dt]= lmdb.open("%s/%s.lmdb" % (self.db_path, dt),
                                        map_size=config.DB_MAPSIZES[dt])
            self.db_entries[dt] = self.db_envs[dt].stat()['entries']
            self.total_entries += self.db_entries[dt]
            self.db_indices[dt] = 0

        for dt in config.DATA_TYPES:
            self.label_ratio[dt] = self.db_entries[dt] / self.total_entries

        self.task_dts_map = {'label': ['pos', 'neg'],
                             'bbox': ['pos', 'part'],
                             'landmark': ['landmark']}
        self.task_dts_ratios =  {'pos': 0.2, 'part': 0.2, 'neg': 0.6}

    def next_batch(self, batch, task):
        # return self.get_data(self.task_dts_map[task], batch)
        return self.get_data(['neg', 'part', 'pos'], batch)

    def get_data(self, dts, batch):
        # total = sum(self.db_entries[dt] for dt in dts)
        # self.ratios = {dt:float(self.db_entries[dt]) / total for dt in dts}
        data = []
        for k, dt in enumerate(dts):
            if k == len(dts) - 1:
                sample_num = batch - len(data)
            else:
                sample_num = max(round(batch * self.task_dts_ratios[dt]), 1)
                # sample_num = max(round(batch * ratios[dt]), 1)
            with self.db_envs[dt].begin() as txn:
                cursor = txn.cursor()
                for i in range(0, sample_num):
                    if self.db_indices[dt] >= self.db_entries[dt]:
                        self.db_indices[dt] = 0
                    cursor.set_key(str(self.db_indices[dt]).encode())
                    datum = mtcnn_pb2.Datum()
                    datum.ParseFromString(cursor.value())
                    img = self.recover_img(datum)
                    bbox = datum.bbox
                    if random.choice([0, 1]) == 1:
                        if dt == 'neg':
                            img = cv2.flip(img, random.choice([-1, 0, 1]))
                        # elif dt == 'pos' or dt == 'part':
                        #     img = cv2.flip(img, 1)
                        #     bbox = [bbox[0], bbox[3], bbox[2], bbox[1]]
                    data.append([img, datum.label, bbox, datum.landm])
                    self.db_indices[dt] += 1
        return data

    def recover_img(self, datum):
        return np.frombuffer(datum.img).reshape((3, self.img_size, self.img_size))


