import config
import lmdb
import random
import mtcnn_pb2

class BatchLoader:
    def __init__(self, param):
        self.net = param['net']
        self.img_size = config.NET_IMG_SIZES[self.net]
        self.db_envs = {}
        self.db_path = os.path.join(config.DB_PATH, self.net)
        self.db_entries = {}

        self.label_ratio[dt] = {}

        self.total_entries = 0
        self.total_labels = 0
        self.total_bboxes = 0
        self.total_landmark = 0
        for dt in config.DATA_TYPES:
            self.db_envs[dt]= lmdb.open("%s/%s.lmdb" % (self.db_path, dt),
                                        map_size=config.DB_MAPSIZES[dt])
            self.db_entries[dt] = self.db_envs[dt].stat()['entries']
            self.total_entries += self.db_entries[dt]

        for dt in config.DATA_TYPES:
            self.label_ratio[dt] = self.db_entries[dt] / self.total_entries

        self.task_dts_map = {'label': ['pos', 'neg'],
                             'bbox': ['pos', 'part'],
                             'landmark': ['landmark']}

    def next_batch(self, task, batch):
        return self.get_data(self.task_dts_map[task], batch)

    def get_data(self, dts, num):
        total = sum(self.db_entries[dt] for dt in dts)
        ratios = {dt:float(self.db_entries[dt]) / total for dt in dts}
        data = []
        for dt in dts:
            sample_num = min(max(int(batch * ratios[dt]), 1), batch - len(batch_data))
            indices = random.sample(range(0, num))
            with self.db_envs[dt].begin() as txn:
                cursor = tnx.cursor()
                for i in indices:
                    cursor.set_key(str(i))
                    datum = mtcnn_pb2.Datum()
                    datum.ParseFromString(cursor.value())
                    data.append(datum)
        return data

    def recover_img(self, img):
        return np.frombuffer(datum.img).reshape((3, self.img_size, self.img_size))


