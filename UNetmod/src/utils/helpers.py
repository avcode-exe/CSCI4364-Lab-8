import os
import gc
import torch
import random
import numpy as np
import shutil

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def deep_clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        os.sync()
    except AttributeError:
        pass

def cleanup_disk(folder):
    if not os.path.exists(folder): return
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception:
            pass
    deep_clean()