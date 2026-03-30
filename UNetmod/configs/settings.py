import os

class Config:
    # Training Hyperparameters
    LR = 5e-5
    EPOCHS = 3
    BATCH_SIZE = 8
    SAVE_EVERY = 1000
    TARGET_FRAME = 300
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cnn_unet_step_checkpoint.pt")
    TEMP_DATA_DIR = "/kaggle/tmp/bubble_data"
    HF_QUOTA_DIR = "/kaggle/tmp/huggingface_quota"
    CACHE_DIR = "/kaggle/tmp/cache"
    
    # Fluid Physics Metadata
    FLUID_METADATA = {
        "FC72": {"t_bulk": 58.0, "t_scale": 60.0, "name": "FC-72"},
        "R515B": {"t_bulk": -19.0, "t_scale": 60.0, "name": "R-515B"},
        "LN2": {"t_bulk": -196.0, "t_scale": 50.0, "name": "LN-2"}
    }

    @classmethod
    def setup_env(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.TEMP_DATA_DIR, exist_ok=True)
        os.environ["HF_HOME"] = cls.HF_QUOTA_DIR
        os.environ["XDG_CACHE_HOME"] = cls.CACHE_DIR