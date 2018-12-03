from batch_loader import BatchLoader

import config
import random

if __name__ == '__main__':
    param = {"net": "pnet", "batch": 64}
    batch_loader = BatchLoader(param)
    task = random.choice(config.TRAIN_TASKS[param['net']])
    data = batch_loader.next_batch(64, task)
    for datum in data:
        print(datum)
