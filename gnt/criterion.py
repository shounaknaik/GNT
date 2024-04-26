import torch
import torch.nn as nn
from utils import img2mse
import numpy as np

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]
        # print(pred_rgb)
        # print(gt_rgb)

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log

class DepthCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, colmap_depth_map):
        """
        training depth criterion
        """
        # Attention depth is between 2 and 6
        pred_depth_attention = outputs["depth"]
        us = ray_batch["u_positions"]
        vs = ray_batch["v_positions"]

        #Colmap_depth_map converting to probablity fn

        #U is the column and v is the row
        colmap_depth = colmap_depth_map[vs,us]

        # loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        # print("Inside Depth Loss")
        
        # print(len(pred_depth_attention))
        # print(len(colmap_depth))
        # print(type(pred_depth_attention))
        # print(type(colmap_depth))

        # Assuming pred_depth_attention and colmap_depth are lists of probabilities
        # pred_depth_attention = torch.tensor(pred_depth_attention)
        colmap_depth = torch.tensor(colmap_depth).to('cuda:0')

        # Ensure probabilities sum up to 1 (normalize)
        pred_depth_attention /= torch.sum(pred_depth_attention)

        # print(pred_depth_attention)
        # print(colmap_depth)

        # Compute KL divergence
        # Ground truth is colmap depth and we want to match the pred_depth_attention to colmap_depth.
        # Thus q is pred_depth_attention and p is colmap_depth
        #Scaling the loss by half to match ranges of loss values from rgb
        # kl_divergence = torch.sum(torch.where(colmap_depth !=0, colmap_depth * torch.log(colmap_depth / pred_depth_attention),0))*0.5

        epsilon = 1e-8  # Small epsilon value to avoid division by zero

        # Compute KL divergence with added epsilon for numerical stability
        kl_divergence = torch.sum(
            torch.where(
                colmap_depth != 0,
                colmap_depth * torch.log((colmap_depth + epsilon) / (pred_depth_attention + epsilon)),
                torch.tensor(0.0)  # Handle zero entries in colmap_depth
            )
        ) * 0.5

        # print("KL Divergence:", kl_divergence)
        # input('q')

        return kl_divergence

