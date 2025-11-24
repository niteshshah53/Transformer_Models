import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk

# Focal Loss for multi-class segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # FIXED: Proper Focal Loss implementation
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction='none')
        else:
            ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss for extreme class imbalance.
    
    Based on "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019).
    Best for extreme imbalance (>100:1 ratio).
    
    Formula: CB_loss = (1 - beta) / (1 - beta^n_i) * CE_loss
    where:
    - beta: hyperparameter (typically 0.9999 for extreme imbalance)
    - n_i: number of samples for class i
    - The re-weighting factor is (1 - beta) / (1 - beta^n_i)
    
    Note: This implementation uses class_weights that are already computed using ENS (Effective
    Number of Samples) formula. The weights are applied per-pixel based on the pixel's class,
    providing fine-grained re-weighting compared to global CE weighting.
    
    Args:
        class_weights: Per-class weights (typically computed using ENS formula from class frequencies)
        num_classes: Number of classes
        beta: Hyperparameter for effective number (default: 0.9999, kept for compatibility but not used if weights are pre-computed)
        ignore_index: Class index to ignore in loss computation
        label_smoothing: Label smoothing factor (default: 0.1)
    """
    def __init__(self, class_weights, num_classes, beta=0.9999, ignore_index=None, label_smoothing=0.1):
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        
        # Use class_weights directly as CB weights
        # Note: class_weights are typically already computed using ENS formula: (1-beta) / (1-beta^n_i)
        # So we can use them directly, but normalize to ensure stability
        if class_weights is not None:
            if isinstance(class_weights, torch.Tensor):
                self.cb_weights = class_weights.clone().detach()
            else:
                self.cb_weights = torch.tensor(class_weights, dtype=torch.float32)
            
            # Normalize CB weights to have mean=1 for stability (prevents loss scale issues)
            mean_weight = self.cb_weights.mean().item()
            if mean_weight > 1e-8:
                self.cb_weights = self.cb_weights / mean_weight
        else:
            # Uniform weights if no class weights provided
            self.cb_weights = torch.ones(num_classes, dtype=torch.float32)
        
        # Move to device when forward is called
        self._device = None
    
    def forward(self, input, target):
        """
        Forward pass of Class-Balanced Loss.
        
        Args:
            input: Model predictions [B, C, H, W] or [B, C]
            target: Ground truth labels [B, H, W] or [B]
        
        Returns:
            Scalar loss value
        """
        # Move weights to same device as input
        if self._device != input.device:
            self.cb_weights = self.cb_weights.to(input.device)
            self._device = input.device
        
        # Compute standard cross-entropy loss per sample
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(
                input, target, 
                weight=None,  # Don't use weight here, we'll apply CB weights manually
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
                reduction='none'
            )
        else:
            ce_loss = F.cross_entropy(
                input, target,
                weight=None,
                label_smoothing=self.label_smoothing,
                reduction='none'
            )
        
        # Apply class-balanced re-weighting per pixel/sample
        # Get class indices for each pixel/sample
        if len(target.shape) == 3:  # [B, H, W] - segmentation
            # Flatten for easier indexing
            target_flat = target.view(-1)  # [B*H*W]
            ce_loss_flat = ce_loss.view(-1)  # [B*H*W]
            
            # Create mask for valid pixels (exclude ignore_index if specified)
            if self.ignore_index is not None:
                valid_mask = (target_flat != self.ignore_index)
            else:
                valid_mask = torch.ones_like(target_flat, dtype=torch.bool)
            
            # Get CB weight for each pixel's class
            # Use 0 weight for ignore_index pixels (they'll be masked out)
            cb_weights_per_pixel = torch.zeros_like(ce_loss_flat)
            if valid_mask.any():
                # Clamp target indices to valid range to avoid index errors
                target_clamped = torch.clamp(target_flat, 0, self.num_classes - 1)
                cb_weights_per_pixel[valid_mask] = self.cb_weights[target_clamped[valid_mask]]
            
            # Apply re-weighting
            cb_loss_flat = ce_loss_flat * cb_weights_per_pixel
            
            # Average over valid pixels only
            if valid_mask.any():
                return cb_loss_flat[valid_mask].mean()
            else:
                return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        else:  # [B] - classification
            # Create mask for valid samples (exclude ignore_index if specified)
            if self.ignore_index is not None:
                valid_mask = (target != self.ignore_index)
            else:
                valid_mask = torch.ones_like(target, dtype=torch.bool)
            
            # Get CB weight for each sample's class
            cb_weights_per_sample = torch.zeros_like(ce_loss)
            if valid_mask.any():
                # Clamp target indices to valid range
                target_clamped = torch.clamp(target, 0, self.num_classes - 1)
                cb_weights_per_sample[valid_mask] = self.cb_weights[target_clamped[valid_mask]]
            
            # Apply re-weighting
            cb_loss = ce_loss * cb_weights_per_sample
            
            # Average over valid samples only
            if valid_mask.any():
                return cb_loss[valid_mask].mean()
            else:
                return torch.tensor(0.0, device=input.device, dtype=input.dtype)
    

class DiceLoss(nn.Module):
    """
    Dice Loss with proper weight handling and NaN prevention.
    
    Fixes from original:
    - Added weight parameter to __init__
    - Increased smooth from 1e-5 to 1e-4 for stability
    - Added NaN detection and handling
    - Fixed normalization (sum of weights instead of n_classes)
    - Prevent division by zero for empty classes
    """
    def __init__(self, n_classes, weight=None, smooth=1e-4):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight  # Store default weight
        self.smooth = smooth  # Increased for numerical stability

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        """
        Compute Dice loss for a single class with NaN prevention.
        """
        target = target.float()
        smooth = self.smooth
        
        # Check for NaN in inputs before computation
        if torch.isnan(score).any() or torch.isnan(target).any():
            # If inputs contain NaN, return 0 loss to avoid propagating NaN
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        # Clamp values to prevent overflow/underflow
        score = torch.clamp(score, min=0.0, max=1.0)
        target = torch.clamp(target, min=0.0, max=1.0)
        
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        union = z_sum + y_sum
        
        # Check for NaN in computed values
        if torch.isnan(intersect) or torch.isnan(union):
            # Return 0 loss if computation produced NaN
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        # CRITICAL FIX: Prevent NaN when both prediction and target are empty
        # If union is extremely small, return 0 loss (perfect for absent class)
        if union < smooth * 10:
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        dice_coef = (2 * intersect + smooth) / (union + smooth)
        loss = 1 - dice_coef
        
        # Safety check for NaN in final loss (shouldn't happen with above fixes)
        if torch.isnan(loss):
            return torch.tensor(0.0, device=score.device, dtype=score.dtype)
        
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        """
        Forward pass with proper weight handling.
        
        Args:
            inputs: Model predictions [B, C, H, W]
            target: Ground truth labels [B, H, W]
            weight: Optional class weights (overrides self.weight if provided)
            softmax: Whether to apply softmax to inputs
        """
        # Check for NaN or Inf in inputs before processing
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            # Replace NaN/Inf with zeros to prevent propagation
            inputs = torch.where(torch.isnan(inputs) | torch.isinf(inputs), 
                                torch.zeros_like(inputs), inputs)
        
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            # Check again after softmax in case it produced NaN
            if torch.isnan(inputs).any():
                inputs = torch.where(torch.isnan(inputs), 
                                    torch.zeros_like(inputs), inputs)
                # Renormalize to ensure valid probability distribution
                inputs = inputs / (inputs.sum(dim=1, keepdim=True) + 1e-8)
        
        target = self._one_hot_encoder(target)
        
        # Determine which weights to use
        if weight is not None:
            # Use provided weight (from forward call)
            if isinstance(weight, torch.Tensor):
                class_weights = weight.cpu().numpy().tolist()
            else:
                class_weights = weight
        elif self.weight is not None:
            # Use weight from __init__
            if isinstance(self.weight, torch.Tensor):
                class_weights = self.weight.cpu().numpy().tolist()
            else:
                class_weights = self.weight
        else:
            # No weights provided, use uniform weights
            class_weights = [1] * self.n_classes
        
        # Size check
        assert inputs.size() == target.size(), \
            f'predict {inputs.size()} & target {target.size()} shape do not match'
        
        # Compute weighted Dice loss per class
        loss = 0.0
        total_weight = 0.0
        
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            
            # Skip NaN losses (shouldn't happen with fixes)
            if not torch.isnan(dice):
                loss += dice * class_weights[i]
                total_weight += class_weights[i]
        
        # CRITICAL FIX: Normalize by sum of weights, not n_classes
        # This properly handles weighted classes
        if total_weight > 0:
            return loss / total_weight
        else:
            # Fallback (should never happen)
            print("⚠️  No valid classes in Dice Loss!")
            return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy().squeeze(0), label.squeeze(0).cpu().detach().numpy().squeeze(0)
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list