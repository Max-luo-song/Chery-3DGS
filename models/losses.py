import numpy as np
from typing import Literal, Union

import torch
import torch.nn.functional as F
from torch import autograd, nn, Tensor

def reduce(
    loss: Union[torch.Tensor, np.ndarray], 
    mask: Union[torch.Tensor, np.ndarray] = None, 
    reduction: Literal['mean', 'mean_in_mask', 'sum', 'max', 'min', 'none']='mean'):

    if mask is not None:
        if mask.dim() == loss.dim() - 1:
            mask = mask.view(*loss.shape[:-1], 1).expand_as(loss)
        assert loss.dim() == mask.dim(), f"Expects loss.dim={loss.dim()} to be equal to mask.dim()={mask.dim()}"
    
    if reduction == 'mean':
        return loss.mean() if mask is None else (loss * mask).mean()
    elif reduction == 'mean_in_mask':
        return loss.mean() if mask is None else (loss * mask).sum() / mask.sum().clip(1e-5)
    elif reduction == 'sum':
        return loss.sum() if mask is None else (loss * mask).sum()
    elif reduction == 'max':
        return loss.max() if mask is None else loss[mask].max()
    elif reduction == 'min':
        return loss.min() if mask is None else loss[mask].min()
    elif reduction == 'none':
        return loss if mask is None else loss * mask
    else:
        raise RuntimeError(f"Invalid reduction={reduction}")

class SafeBCE(autograd.Function):
    """ Perform clipped BCE without disgarding gradients (preserve clipped gradients)
        This function is equivalent to torch.clip(x, limit), 1-limit) before BCE, 
        BUT with grad existing on those clipped values.
        
    NOTE: pytorch original BCELoss implementation is equivalent to limit = np.exp(-100) here.
        see doc https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    """
    @staticmethod
    def forward(ctx, x, y, limit):
        assert (torch.where(y!=1, y+1, y)==1).all(), u'target must all be {0,1}'
        ln_limit = ctx.ln_limit = np.log(limit)
        # ctx.clip_grad = clip_grad
        
        # NOTE: for example, torch.log(1-torch.tensor([1.000001])) = nan
        x = torch.clip(x, 0, 1)
        y = torch.clip(y, 0, 1)
        ctx.save_for_backward(x, y)
        return -torch.where(y==0, torch.log(1-x).clamp_min_(ln_limit), torch.log(x).clamp_min_(ln_limit))
        # return -(y * torch.log(x).clamp_min_(ln_limit) + (1-y)*torch.log(1-x).clamp_min_(ln_limit))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        ln_limit = ctx.ln_limit
        
        # NOTE: for y==0, do not clip small x; for y==1, do not clip small (1-x)
        limit = np.exp(ln_limit)
        # x = torch.clip(x, eclip, 1-eclip)
        x = torch.where(y==0, torch.clip(x, 0, 1-limit), torch.clip(x, limit, 1))
        
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            # ttt = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # with open('grad.txt', 'a') as fp:
            #     fp.write(f"{ttt.min().item():.05f}, {ttt.max().item():.05f}\n")
            # NOTE: " * (~(x==y))" so that those already match will not generate gradients.
            grad_x = torch.where(y==0, 1/(1-x), -1/x) * grad_output * (~(x==y))
            # grad_x = ( (1-y)/(1-x) - y/x ) * grad_output
        if ctx.needs_input_grad[1]:
            grad_y = (torch.log(1-x) - torch.log(x)) * grad_output * (~(x==y))
        #---- x, y, limit
        return grad_x, grad_y, None

def safe_binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, limit: float = 0.1, reduction="mean") -> torch.Tensor:
    loss = SafeBCE.apply(input, target, limit)
    return reduce(loss, None, reduction=reduction)

def binary_cross_entropy(input: torch.Tensor, target: torch.Tensor, reduction="mean") -> torch.Tensor:
    loss = F.binary_cross_entropy(input, target, reduction="none")
    return reduce(loss, None, reduction=reduction)

def normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 0.0, 1.0)

def safe_normalize_depth(depth: Tensor, max_depth: float = 80.0):
    return torch.clamp(depth / max_depth, 1e-06, 1.0)

class DepthLoss(nn.Module):
    def __init__(
        self,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l2",
        normalize: bool = True,
        use_inverse_depth: bool = False,
        depth_error_percentile: float = None,
        upper_bound: float = 80,
        reduction: Literal["mean_on_hit", "mean_on_hw", "sum", "none"] = "mean_on_hit",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.use_inverse_depth = use_inverse_depth
        self.upper_bound = upper_bound
        self.depth_error_percentile = depth_error_percentile
        self.reduction = reduction

    def _compute_depth_loss(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        max_depth: float = 80,
        hit_mask: Tensor = None,
    ):
        pred_depth = pred_depth.squeeze()
        gt_depth = gt_depth.squeeze()
        if hit_mask is not None:
            pred_depth = pred_depth * hit_mask
            gt_depth = gt_depth * hit_mask
        
        # cal valid mask to make sure gt_depth is valid
        valid_mask = (gt_depth > 0.01) & (gt_depth < max_depth) & (pred_depth > 0.0001)
        
        # normalize depth to (0, 1)
        if self.normalize:
            pred_depth = safe_normalize_depth(pred_depth[valid_mask], max_depth=max_depth)
            gt_depth = safe_normalize_depth(gt_depth[valid_mask], max_depth=max_depth)
        else:
            pred_depth = pred_depth[valid_mask]
            gt_depth = gt_depth[valid_mask]
        
        # inverse the depth map (0, 1) -> (1, +inf)
        if self.use_inverse_depth:
            pred_depth = 1./pred_depth
            gt_depth = 1./gt_depth
            
        # cal loss
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(pred_depth, gt_depth, reduction="none")
        elif self.loss_type == "l2":
            return F.mse_loss(pred_depth, gt_depth, reduction="none")
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def __call__(
        self,
        pred_depth: Tensor,
        gt_depth: Tensor,
        hit_mask: Tensor = None,
    ):
        depth_error = self._compute_depth_loss(pred_depth, gt_depth, self.upper_bound, hit_mask)
        if self.depth_error_percentile is not None:
            # to avoid outliers. not used for now
            depth_error = depth_error.flatten()
            depth_error = depth_error[
                depth_error.argsort()[
                    : int(len(depth_error) * self.depth_error_percentile)
                ]
            ]
        
        if self.reduction == "sum":
            depth_error = depth_error.sum()
        elif self.reduction == "none":
            depth_error = depth_error
        elif self.reduction == "mean_on_hit":
            depth_error = depth_error.mean()
        elif self.reduction == "mean_on_hw":
            n = gt_depth.shape[0]*gt_depth.shape[1]
            depth_error = depth_error.sum() / n
        else:
            raise NotImplementedError(f"Unknown reduction method: {self.reduction}")

        return depth_error


def quaternion_to_euler(quat: torch.Tensor):
    """
    将四元数转换为欧拉角（roll, pitch, yaw）
    输入: 四元数张量 (N, 4)，格式为 [w, x, y, z]
    输出: roll, pitch, yaw 张量 (N,)
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # 计算roll (x轴旋转)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    
    # 计算pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    # 防止数值不稳定
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    # 计算yaw (z轴旋转)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    return roll, pitch, yaw

### 朝向lossv1版本：限制高斯朝向roll，pitch, scale_z
class RoadOrientationLossV1(nn.Module):
    def __init__(self, roll_limit: float = 0.3, pitch_limit: float = 0.3, 
                 vertical_scale_limit: float = 0.5):
        super(RoadOrientationLossV1, self).__init__()
        self.roll_limit = roll_limit
        self.pitch_limit = pitch_limit
        self.vertical_scale_limit = vertical_scale_limit
        
    def forward(self, road_quats: torch.Tensor, road_scales: torch.Tensor) -> torch.Tensor:
        """
        计算朝向约束损失
        """
        # 将四元数转换为欧拉角
        roll, pitch, yaw = quaternion_to_euler(road_quats)
        
        # 计算各角度的绝对值损失
        roll_loss = torch.abs(roll) / self.roll_limit
        pitch_loss = torch.abs(pitch) / self.pitch_limit
        # 2. 获取scale的z轴分量（vertical scale）
        scale_z = road_scales[:, 2]  # 假设scales的顺序为[sx, sy, sz]，sz是第三个维度
        vertical_scale_loss = torch.abs(scale_z) / self.vertical_scale_limit  # 限制垂直拉伸
        # 只对road和sky应用约束
        orientation_loss = torch.mean(roll_loss + pitch_loss + vertical_scale_loss)
        
        return orientation_loss

def quaternion_to_up_vector(quat: torch.Tensor) -> torch.Tensor:
    """
    将四元数转换为朝上向量（即z轴方向）
    输入: 四元数张量 (N, 4)，格式为 [w, x, y, z]
    输出: 朝上向量 (N, 3)，主要是z分量
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # 旋转后的z轴方向向量
    # 初始z轴(0,0,1)经过四元数旋转后的结果
    up_x = 2 * (x * z + w * y)
    up_y = 2 * (y * z - w * x)
    up_z = 1 - 2 * (x * x + y * y)
    
    return torch.stack([up_x, up_y, up_z], dim=-1)

### 朝向lossv2版本：路面都朝上，接近法线(0,0,1)
class RoadOrientationLossV2(nn.Module):
    def __init__(self, beta: float = 0.1, up_vector_weight: float = 1.0, vertical_scale_limit: float = 0.5, roll_limit: float = 0.3, pitch_limit: float = 0.3):
        super(RoadOrientationLossV2, self).__init__()
        self.beta = beta
        self.up_vector_weight = up_vector_weight
        self.vertical_scale_limit = vertical_scale_limit
        self.roll_limit = roll_limit
        self.pitch_limit = pitch_limit

    def forward(self, road_quats: torch.Tensor, road_scales: torch.Tensor) -> torch.Tensor:
        """
        计算朝向约束损失
        """
        # # 将四元数转换为朝上向量
        up_vectors = quaternion_to_up_vector(road_quats)        
        # 计算朝上向量的z分量（应该接近1）
        up_z = up_vectors[:, 2]
        # 计算x和y分量（应该接近0）
        up_xy = torch.sqrt(up_vectors[:, 0]**2 + up_vectors[:, 1]**2)
        up_loss = torch.mean(1 - up_z) + torch.mean(up_xy)
        return self.beta * self.up_vector_weight * up_loss

### NOTE(gls): 朝向loss，在输入中增加朝向输入
class RoadLoss(nn.Module):
    def __init__(self, version, lambda_dssim: float = 0.5, beta: float = 0.1, 
                 up_vector_weight: float = 1.0):
        super(RoadLoss, self).__init__() 
        self.lambda_dssim = lambda_dssim
        if version == 1:
            self.road_orientation_loss = RoadOrientationLossV1()
        elif version == 2:
            self.road_orientation_loss = RoadOrientationLossV2()
    def _compute_road_loss(self, pred_road_rgb: torch.Tensor, gt_road_rgb: torch.Tensor) -> torch.Tensor:
        Ll1 = torch.abs(gt_road_rgb - pred_road_rgb).mean()
        return Ll1
    def __call__(
        self,
        pred_road_rgb: Tensor,
        gt_road_rgb: Tensor,
        road_quats: Tensor,
        road_scales: Tensor
    ):
        road_loss = self._compute_road_loss(pred_road_rgb, gt_road_rgb)
        # 计算朝向约束损失（这会通过梯度影响road_quats）
        orientation_loss = self.road_orientation_loss(road_quats, road_scales)
        total_loss = (1 - self.lambda_dssim) * road_loss + \
                    self.lambda_dssim * orientation_loss
        return total_loss