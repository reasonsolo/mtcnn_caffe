NETS = ['pnet', 'rnet', 'onet']
NET_IMG_SIZES = {'pnet': 12, 'rnet':24, 'onet': 48}
NET_OUTPUTS = {
    'pnet': {'bbox': 'conv4-2', 'label': 'prob1'},
    'rnet': {'bbox': 'conv5-2', 'label': 'prob1', 'landm5': 'conv5-3'},
    'onet': {'bbox': 'conv6-2', 'label': 'prob1', 'landm5': 'conv6-3'},
}

DATA_TYPES = {'pos': 1, 'neg': 0, 'part': 2, 'landm5': 3}
DATA_TYPE_RATIOS = {
    'pnet': {'pos': 0.2, 'part': 0.2, 'neg': 0.6},  # suppress landm5 for pnet
    'rnet': {'pos': 0.1, 'part': 0.15, 'neg': 0.55, 'landm5': 0.1},
    'rnet': {'pos': 0.1, 'part': 0.15, 'neg': 0.5, 'landm5': 0.15},
}
MAX_EXAMPLES = {'pos': 12, 'neg': 50, 'part': 24, 'landm5': 20}

LANDMARK_SIZE = 5

DATA_DIR='data_%s'
DB_DIR='db_%s'
DB_PATH='db_train'
DB_MAPSIZES = {'pos': 2 * 1024 * 1024 * 1024,
            'part': 6 * 1024 * 1024 * 1024,
            'neg': 10 * 1024 * 1024 * 1024,
            'landm5': 5 * 1024 * 1024 * 1024,}

MIN_IMG_SIZE = 48

TRAIN_DATA_DIR = 'data_%s'
TRAIN_TASKS = {
    'pnet': ['label', 'bbox'],
    'rnet': ['label', 'bbox', 'landmark'],
    'onet': ['label', 'bbox', 'landmark'],
}


WIDER_DIR  = './'
LFW_DIR    = './'
CELEBRA_DIR= './'
W300_DIR   = '/data/300W-Convert/300W-Original/'

MODEL_DIR = './models/'
TEST_ITER = 300000
