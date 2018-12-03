NETS = ['pnet', 'rnet', 'onet']
NET_IMG_SIZES = {'pnet': 12, 'rnet':24, 'onet': 48}

DATA_TYPES = {'pos': 1, 'neg': 0, 'part': 2, 'landmark': 3}

LANDMARK_SIZE = 5

DB_PATH='db_train'
DB_MAPSIZES = {'pos': 2 * 1024 * 1024 * 1024,
            'part': 6 * 1024 * 1024 * 1024,
            'neg': 10 * 1024 * 1024 * 1024,
            'landmark': 5 * 1024 * 1024 * 1024,}

TRAIN_DATA_DIR = 'data_train'
TRAIN_TASKS = {
    'pnet': ['label', 'bbox'],
    'rnet': ['label', 'bbox', 'landmark'],
    'onet': ['label', 'bbox', 'landmark'],
}

WIDER_DIR='./'
