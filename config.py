NETS = ['pnet', 'rnet', 'onet']
NET_IMG_SIZES = {'pnet': 12, 'rnet':24, 'onet': 48}

DATA_TYPES = {'pos': 1, 'neg': -1, 'part': 0, 'landmark': 2}

LANDMARK_SIZE = 5

DB_PATH='db'
DB_SIZES = {'pos': 1024 * 1024 * 1024, 'part': 5 * 1024 * 1024 * 1024, 'neg': 10 * 1024 * 1024 * 1024}

TRAIN_TASKS = ['label', 'bbox', 'landmark']
