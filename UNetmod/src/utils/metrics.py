import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy

def get_physics_metrics(pred, target):
    """
    Calculates Relative L2 and Eikonal Loss
    pred/target: [B, 4, H, W]
    """
    eps = 1e-2
    # Rel L2
    rel_l2 = torch.norm(pred - target, p=2) / (torch.norm(target, p=2) + eps)
    
    # Eikonal (on SDF channel 3)
    phi = pred[:, 3:4]
    dy = (phi[:, :, 2:, :] - phi[:, :, :-2, :]) / 2.0
    dx = (phi[:, :, :, 2:] - phi[:, :, :, :-2]) / 2.0
    grad_mag = torch.sqrt(dx[:, :, 1:-1, :]**2 + dy[:, :, :, 1:-1]**2 + 1e-8)
    eikonal = torch.abs(grad_mag - 0.1).mean()
    
    return rel_l2.item(), eikonal.item()

def get_heat_flux_kl(pred, target):
    # Gradient at bottom wall
    q_pred = (pred[0, 0, 1, :] - pred[0, 0, 0, :]).cpu().numpy().flatten()
    q_gt = (target[0, 0, 1, :] - target[0, 0, 0, :]).cpu().numpy().flatten()
    
    p_pred, _ = np.histogram(q_pred, bins=50, density=True)
    p_gt, _ = np.histogram(q_gt, bins=50, density=True)
    
    return entropy(p_gt + 1e-6, p_pred + 1e-6)