import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_download
from configs.settings import Config
from src.utils.helpers import cleanup_disk

class BubbleFastDataset(IterableDataset):
    def __init__(self, file_list, start_frame=50):
        self.file_list = file_list
        self.start_frame = start_frame

    def __iter__(self):
        for file_path in self.file_list:
            local_path = hf_hub_download(
                repo_id="hpcforge/BubbleML_2",
                filename=file_path,
                repo_type="dataset",
                local_dir=Config.TEMP_DATA_DIR,
                local_dir_use_symlinks=False
            )
            try:
                with h5py.File(local_path, 'r') as h5:
                    t = h5['temperature'][()]
                    ux = h5['velx'][()]
                    uy = h5['vely'][()]
                    phi = h5['dfun'][()]

                fluid_key = "LN2" if "LN2" in file_path else "FC72" if "FC72" in file_path else "R515B"
                meta = Config.FLUID_METADATA[fluid_key]

                for i in range(self.start_frame, t.shape[0] - 2):
                    tx = torch.from_numpy(np.stack([t[i], ux[i], uy[i], phi[i]])).float()
                    ty = torch.from_numpy(np.stack([t[i+1], ux[i+1], uy[i+1], phi[i+1]])).float()
                    
                    # Norm
                    tx[0] = (tx[0] - meta["t_bulk"]) / meta["t_scale"]; tx[1:4] /= 10.0
                    ty[0] = (ty[0] - meta["t_bulk"]) / meta["t_scale"]; ty[1:4] /= 10.0
                    yield tx, ty
            finally:
                cleanup_disk(Config.TEMP_DATA_DIR)