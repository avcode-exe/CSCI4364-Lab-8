import argparse
import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from config.settings import Config
from src.models.unet_cnn import UNetmod
from src.utils.metrics import get_physics_metrics, get_heat_flux_kl
from src.utils.helpers import deep_clean

def evaluate_frame(model_path, file_path, target_frame):
    """
    Downloads a specific file, predicts a frame, calculates metrics, and plots results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading model from {model_path}...")
    model = UNetmod().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please train the model first.")
        return
    model.eval()

    # 2. Download File from HuggingFace
    print(f"Downloading {file_path} from BubbleML_2 repo...")
    local_path = hf_hub_download(
        repo_id="hpcforge/BubbleML_2",
        filename=file_path,
        repo_type="dataset",
        local_dir=Config.TEMP_DATA_DIR,
        local_dir_use_symlinks=False
    )

    try:
        with h5py.File(local_path, 'r') as h5:
            # Check if frame index is valid
            total_frames = h5['temperature'].shape[0]
            if target_frame >= total_frames or target_frame < 1:
                print(f"Error: Target frame {target_frame} is out of bounds (1-{total_frames-1}).")
                return

            idx_in, idx_out = target_frame - 1, target_frame
            t = h5['temperature'][[idx_in, idx_out]]
            ux = h5['velx'][[idx_in, idx_out]]
            uy = h5['vely'][[idx_in, idx_out]]
            phi = h5['dfun'][[idx_in, idx_out]]

        # Determine fluid metadata for normalization based on filename
        fluid_key = "LN2" if "LN2" in file_path else "FC72" if "FC72" in file_path else "R515B"
        meta = Config.FLUID_METADATA[fluid_key]

        # 3. Prepare Tensors & Normalize (Paper Style: T = (T-Bulk)/Scale, others / 10)
        tx = torch.from_numpy(np.stack([t[0], ux[0], uy[0], phi[0]])).float().unsqueeze(0).to(device)
        ty = torch.from_numpy(np.stack([t[1], ux[1], uy[1], phi[1]])).float().unsqueeze(0).to(device)
        
        tx[:, 0] = (tx[:, 0] - meta["t_bulk"]) / meta["t_scale"]
        tx[:, 1:4] /= 10.0
        ty[:, 0] = (ty[:, 0] - meta["t_bulk"]) / meta["t_scale"]
        ty[:, 1:4] /= 10.0

        # 4. Predict & Calculate Metrics
        print(f"Generating prediction for frame {target_frame}...")
        with torch.no_grad():
            pred = model(tx)
        
        rel_l2, eikonal = get_physics_metrics(pred, ty)
        kl_div = get_heat_flux_kl(pred, ty)
        
        # 5. Visualization Plotting
        p_img, g_img = pred[0].cpu().numpy(), ty[0].cpu().numpy()
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        titles = ["Temperature", "Velocity X", "Velocity Y", "SDF (Bubble)"]
        cmaps = ["magma", "viridis", "viridis", "RdBu_r"]

        for c in range(4):
            # Row 0: Ground Truth
            im0 = axes[0, c].imshow(g_img[c], cmap=cmaps[c], origin='lower')
            axes[0, c].set_title(f"GT: {titles[c]}")
            plt.colorbar(im0, ax=axes[0, c])
            
            # Row 1: Prediction
            im1 = axes[1, c].imshow(p_img[c], cmap=cmaps[c], origin='lower')
            axes[1, c].set_title(f"Pred: {titles[c]}")
            plt.colorbar(im1, ax=axes[1, c])

        axes[0,0].set_ylabel("GROUND TRUTH", fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel("UNET PREDICTION", fontsize=12, fontweight='bold')

        info_text = (f"File: {file_path} | Frame: {target_frame}\n"
                     f"RelL2: {rel_l2:.4f} | Eikonal: {eikonal:.4f} | Heat Flux KL: {kl_div:.4f}")
        plt.suptitle(info_text, fontsize=16)
        
        save_name = f"eval_{fluid_key}_frame{target_frame}.png"
        plt.tight_layout()
        plt.savefig(save_name)
        print(f"Results saved to {save_name}")
        plt.show()

    finally:
        deep_clean()

if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Evaluate Bubbleformer CNN-UNet on a specific frame.")
    
    parser.add_name = parser.add_argument(
        "--model", type=str, default="final_model.pt", help="Path to trained model weights"
    )
    parser.add_argument(
        "--file", type=str, required=True, 
        help="HDF5 filename (e.g., 'PoolBoiling-Saturated-FC72-2D/Twall_96.hdf5')"
    )
    parser.add_argument(
        "--frame", type=int, default=500, help="Target frame index to predict"
    )

    args = parser.parse_args()

    # Execute
    evaluate_frame(args.model, args.file, args.frame)