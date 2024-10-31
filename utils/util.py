
import os.path
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

class SetLogger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        directory_path = os.path.dirname(filepath)
        os.makedirs(directory_path, exist_ok=True)

        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def info(self, s):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(s + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()

def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_config(config_path):

    with open(config_path, 'r') as f:

        config = json.load(f)
    return config