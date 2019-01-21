import config
import os
import cv2
import lmdb
import random
import landmark_pb2
import numpy as np

class BatchLoader:
    def __init__(self, param, data_cat):
        self.img_size = config.IMG_SIZE
        self.data_cat = data_cat
        self.db_envs = {}
        self.db_path = os.path.join(config.DB_DIR % self.data_cat)

        self.entry_index = 0
        self.total_entries  = 0
        try:
            self.db_env = lmdb.open("%s" % (self.db_path), map_size=config.DB_MAPSIZE)
            self.total_entries = self.db_env.stat()['entries']
        except:
            pass

    def is_loaded(self):
        return self.total_entries > 0

    def next_batch(self, batch, task):
        # return self.get_data(batch)
        return self.get_random_data(batch)

    def get_data(self, batch_num):
        # total = sum(self.db_entries[dt] for dt in dts)
        # self.ratios = {dt:float(self.db_entries[dt]) / total for dt in dts}
        batch_num = int(batch_num)
        data = []
                # sample_num = max(round(batch * ratios[dt]), 1)
        with self.db_env.begin() as txn:
            cursor = txn.cursor()
            # for i in range(0, batch_num):
            while len(data) < batch_num:
                cursor.set_key(str(self.entry_index % self.total_entries).encode())
                self.entry_index += 1
                datum = landmark_pb2.Datum()
                try:
                    datum.ParseFromString(cursor.value())
                    img = self.recover_img(datum)
                    data.append([img, datum.pts, datum.eye_dist])
                except:
                    continue
        np.random.shuffle(data)
        return data

    def get_random_data(self, batch_num):
        batch_num = int(batch_num)
        data = []
        rand_indices = random.sample(xrange(0, self.total_entries), batch_num)
        with self.db_env.begin() as txn:
            cursor = txn.cursor()
            for i in rand_indices:
                cursor.set_key(str(i % self.total_entries).encode())
                datum = landmark_pb2.Datum()
                try:
                    datum.ParseFromString(cursor.value())
                    img = self.recover_img(datum)
                    data.append([img, datum.pts, datum.eye_dist])
                except:
                    continue
        return data

    def recover_img(self, datum):
        return np.frombuffer(datum.img).reshape((3, self.img_size, self.img_size))


