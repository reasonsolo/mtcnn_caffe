import mtcnn.config
import lmdb

class BatchLoader:
    def __init__(self, lmdb_path):
        self.db_envs = {}
        for dt in config.DATA_TYPES:
            self.db_envs[dt]= lmdb.open("%s/%s.lmdb" % (config.DB_PATH, dt),
                                        config.DB_MAPSIZES[dt])



