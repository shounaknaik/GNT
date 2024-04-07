import torch.nn as nn
from utils import img2mse


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

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log

class DepthCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, colmap_depth_map):
        """
        training depth criterion
        """
        # Attention depth is between 2 and 9
        pred_depth_attention = outputs["depth"]
        us = ray_batch["u_positions"]
        vs = ray_batch["v_positions"]

        colmap_depth = colmap_depth_map[us,vs]

        # loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        print(pred_depth_attention)
        print(colmap_depth)

        input('q')

        return None

