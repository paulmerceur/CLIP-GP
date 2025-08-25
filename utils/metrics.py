"""
Evaluation metrics for CLIP-GP.
"""
import torch
import numpy as np
from typing import Tuple, List


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[float]:
    """
    Compute classification accuracy.
    
    Args:
        logits: Model predictions of shape [N, C]
        labels: Ground truth labels of shape [N]
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracies
    """
    maxk = max(topk)
    batch_size = labels.size(0)
    
    if batch_size == 0:
        return [0.0] * len(topk)
    
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    
    return res

def compute_macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute macro F1 score.
    
    Args:
        logits: Model predictions of shape [N, C]
        labels: Ground truth labels of shape [N]
        
    Returns:
        Macro F1 score
    """
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        raise ImportError("sklearn is required for F1 score computation")
    
    pred = logits.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1 = f1_score(labels_np, pred, average='macro')
    return float(f1 * 100)

def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Args:
        logits: [N, C] model outputs (unnormalized)
        labels: [N] true labels
        n_bins: number of bins
    Returns:
        ECE value (float, 0-1)
    """
    import torch.nn.functional as F
    device = logits.device
    probs = F.softmax(logits, dim=-1)
    conf, preds = probs.max(dim=-1)
    acc = preds.eq(labels).float() 
    ece = torch.zeros(1, device=device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
    for i in range(n_bins):
        in_bin = (conf > bin_boundaries[i]) * (conf <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = acc[in_bin].mean()
            avg_conf_in_bin = conf[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def compute_aece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
    """
    Adaptive Expected Calibration Error (AECE).
    Bins are formed by equal-frequency quantiles of confidence (i.e., each bin
    contains approximately the same number of samples). Returns a value in [0, 1].

    Args:
        logits: [N, C] model outputs (unnormalized)
        labels: [N] true labels
        n_bins: number of adaptive bins
    Returns:
        AECE value (float, 0-1)
    """
    import torch.nn.functional as F
    if logits.numel() == 0:
        return 0.0
    device = logits.device
    probs = F.softmax(logits, dim=-1)
    conf, preds = probs.max(dim=-1)
    acc = preds.eq(labels).float()

    num_samples = conf.numel()
    if num_samples == 0:
        return 0.0
    n_bins = max(1, min(int(n_bins), int(num_samples)))

    # Sort by confidence and split into contiguous bins with ~equal counts
    sorted_conf, sort_idx = torch.sort(conf)
    sorted_acc = acc[sort_idx]

    # Bin edges as indices into the sorted arrays
    edges = torch.linspace(0, num_samples, n_bins + 1, device=device)
    edges = edges.round().long()
    edges[0] = 0
    edges[-1] = num_samples

    aece = torch.zeros(1, device=device)
    for i in range(n_bins):
        left = int(edges[i].item())
        right = int(edges[i + 1].item())
        if right <= left:
            continue
        bin_conf = sorted_conf[left:right]
        bin_acc = sorted_acc[left:right]
        prop_in_bin = (right - left) / float(num_samples)
        avg_conf_in_bin = bin_conf.mean()
        accuracy_in_bin = bin_acc.mean()
        aece += torch.abs(avg_conf_in_bin - accuracy_in_bin) * prop_in_bin

    return float(aece.item())


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.
        
        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class MetricMeter:
    """
    Tracks multiple metrics with running averages.
    """
    
    def __init__(self, delimiter: str = " "):
        self.meters = {}
        self.delimiter = delimiter
    
    def add_meter(self, name: str, meter: AverageMeter):
        """Add a new meter"""
        self.meters[name] = meter
    
    def update(self, **kwargs):
        """Update multiple meters at once"""
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __getattr__(self, name: str):
        """Allow direct access to meters"""
        if name in self.meters:
            return self.meters[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __str__(self):
        """String representation of all meters"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        """Placeholder for distributed training (not implemented)"""
        pass
