import os
import argparse
import torch
from typing import List

from clip import clip
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT
from utils.config import get_cfg_default
from utils.dataset_base import TorchDatasetWrapper, build_dataset


@torch.no_grad()
def encode_texts(model, device, texts: List[str]) -> torch.Tensor:
    tokens = clip.tokenize(texts).to(device)
    feats = model.encode_text(tokens).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


@torch.no_grad()
def encode_images(model, device, images: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(images.to(device).type(model.dtype)).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="val-weighted", choices=["val-weighted", "top-3", "minmax"], help="Template weighting method for building prototypes")
    args = parser.parse_args()
    method = args.method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP
    model_name = "RN50"
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # Build Caltech101 dataset (uses default root from config.py; override via env ROOT if needed)
    cfg = get_cfg_default()
    cfg.dataset.name = "Caltech101"
    #cfg.dataset.name = "OxfordPets"
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        cfg.dataset.root = env_root

    dataset = build_dataset(cfg)
    classnames = dataset.classnames

    # Prepare loaders
    test_ds = TorchDatasetWrapper(dataset.test, transform=preprocess, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    # Validation split for computing weights (fallback to test if missing)
    val_split = getattr(dataset, "val", None)
    if val_split is None:
        val_split = getattr(dataset, "val_x", None)
    if val_split is None:
        print("[WARN] No validation split found; falling back to test for weight computation")
        val_split = dataset.test
    val_ds = TorchDatasetWrapper(val_split, transform=preprocess, is_train=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # Precompute per-class text features for each template separately
    # For each template, we compute zero-shot accuracy using only that template
    print(f"Device: {device}")
    print(f"Classes: {len(classnames)}; Templates: {len(IMAGENET_TEMPLATES_SELECT)}")

    per_template_text_features = []  # list of [num_classes, D]
    for tmpl in IMAGENET_TEMPLATES_SELECT:
        texts = [tmpl.format(c) for c in classnames]
        txt_feats = encode_texts(model, device, texts)
        per_template_text_features.append(txt_feats)

    # Stack text features for convenience
    # per_template_text_features: list of M tensors [K, D]
    M = len(per_template_text_features)
    K = per_template_text_features[0].shape[0] if M > 0 else 0
    D = per_template_text_features[0].shape[1] if M > 0 else 0
    text_feats_kmd = torch.stack(per_template_text_features, dim=0).permute(1, 0, 2).contiguous()  # [K, M, D]

    # Evaluate per-template
    print("Per-template zero-shot accuracies on test:")
    per_template_acc = []
    for tmpl, txt_feats in zip(IMAGENET_TEMPLATES_SELECT, per_template_text_features):
        correct = 0
        total = 0
        for batch in test_loader:
            images = batch["img"]
            labels = batch["label"].to(device)
            img_feats = encode_images(model, device, images)
            logits = img_feats @ txt_feats.t()  # [B, C]
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
        acc = 100.0 * correct / max(total, 1)
        per_template_acc.append(acc)
        print(f"  {acc:6.2f}%  | {tmpl}")

    # Baseline: average templates into one prototype per class, evaluate on test
    avg_txt = torch.stack(per_template_text_features, dim=0).mean(dim=0)  # [K, D]
    correct, total = 0, 0
    for batch in test_loader:
        images = batch["img"]
        labels = batch["label"].to(device)
        img_feats = encode_images(model, device, images)
        logits = img_feats @ avg_txt.t()  # [B, C]
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    avg_acc = 100.0 * correct / max(total, 1)
    print(f"Average over templates (uniform) accuracy on test: {avg_acc:.2f}%")

    # Val-weighted per-class template weights (sanity check; not zero-shot wrt val)
    # Compute weights_raw[k, m] = mean true-class probability using template m on val
    weights_raw = torch.zeros(K, M, dtype=torch.float32, device=device)
    class_counts = torch.zeros(K, dtype=torch.float32, device=device)
    # Pre-move text feats used per template for faster compute
    per_tmpl_txt_on_dev = [t.to(device) for t in per_template_text_features]
    for batch in val_loader:
        images = batch["img"]
        labels = batch["label"].to(device)
        img_feats = encode_images(model, device, images)  # [B, D], normalized
        labels_i64 = labels.to(torch.int64)
        # Update class counts for this minibatch
        binc = torch.bincount(labels_i64, minlength=K).to(torch.float32)
        class_counts += binc
        # For each template, accumulate per-class sum of true-class probabilities
        for m_idx, txt_feats in enumerate(per_tmpl_txt_on_dev):
            logits = img_feats @ txt_feats.t()  # [B, C]
            probs = torch.softmax(logits, dim=-1)  # [B, C]
            true_probs = probs[torch.arange(probs.size(0), device=device), labels_i64]  # [B]
            sums_per_class = torch.zeros(K, dtype=torch.float32, device=device)
            sums_per_class.index_add_(0, labels_i64, true_probs)
            weights_raw[:, m_idx] += sums_per_class
    # Avoid div by zero and normalize per class across templates
    class_counts = class_counts.clamp_min(1.0)
    weights_raw = weights_raw / class_counts.view(-1, 1)  # [K, M]
    weights_raw = torch.clamp(weights_raw, min=0.0)

    # Build final per-class weights based on method
    if method == "top-3":
        # Select top-3 templates overall using average over classes (pre-normalization)
        avg_over_classes_raw = weights_raw.mean(dim=0)  # [M]
        top_k = min(3, M)
        top_vals, top_idx = torch.topk(avg_over_classes_raw, k=top_k, largest=True)
        keep_mask = torch.zeros(M, dtype=torch.float32, device=device)
        keep_mask[top_idx] = 1.0
        # Keep only selected templates; retain their raw magnitudes
        weights_masked = weights_raw * keep_mask.view(1, -1)  # [K, M]
        # For classes with zero sum after masking, fallback to uniform over selected
        denom = weights_masked.sum(dim=1, keepdim=True)
        zero_rows = denom.squeeze(1) <= 1e-12
        if zero_rows.any():
            num_sel = float(top_k)
            # Set selected templates to 1/num_sel for those classes
            uniform_sel = (keep_mask / num_sel).view(1, -1).expand(zero_rows.sum(), -1)
            weights_masked[zero_rows] = uniform_sel
            denom = weights_masked.sum(dim=1, keepdim=True)
        # Normalize per class across selected templates
        weights_km = weights_masked / denom.clamp_min(1e-12)
    elif method == "minmax":
        # Per class: rescale to [0,1] using min-max on raw weights, then renormalize to sum 1
        w = weights_raw.clone()
        w_min = w.min(dim=1, keepdim=True).values
        w_max = w.max(dim=1, keepdim=True).values
        range_ = (w_max - w_min)
        # If range is zero, fall back to uniform
        mask_zero = range_.le(1e-12)
        w = torch.where(mask_zero, torch.full_like(w, 1.0 / float(M)), (w - w_min) / range_.clamp_min(1e-12))
        # Renormalize per class
        weights_km = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
        top_vals = top_idx = None  # not used
    else:
        # Old method: simple per-class normalization of raw weights
        weights_km = weights_raw / weights_raw.sum(dim=1, keepdim=True).clamp_min(1e-12)
        top_vals = top_idx = None

    # Print averaged (over classes) template weights: M numbers
    avg_w_over_classes = weights_km.mean(dim=0).detach().cpu().tolist()
    avg_w_str = ", ".join(f"{v:.4f}" for v in avg_w_over_classes)
    print(f"{method} avg template weights (over classes): [{avg_w_str}]")
    # Also print selected template names in order when using top-3
    if method == "top-3" and top_idx is not None and top_vals is not None:
        selected_templates = [IMAGENET_TEMPLATES_SELECT[i] for i in top_idx.detach().cpu().tolist()]
        print("Selected top-3 templates (val avg score):")
        for rank, (tmpl, val) in enumerate(zip(selected_templates, top_vals.detach().cpu().tolist()), start=1):
            print(f"  {rank}. {val:.4f} | {tmpl}")

    # Build weighted prototypes per class and evaluate on test
    # text_feats_kmd: [K, M, D]; weights_km: [K, M]
    weighted_proto_kd = torch.einsum("km,kmd->kd", weights_km.to(text_feats_kmd.device), text_feats_kmd.to(weights_km.device).to(text_feats_kmd.dtype))
    correct, total = 0, 0
    for batch in test_loader:
        images = batch["img"]
        labels = batch["label"].to(device)
        img_feats = encode_images(model, device, images)
        logits = img_feats @ weighted_proto_kd.to(device).t()  # [B, C]
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    weighted_acc = 100.0 * correct / max(total, 1)
    print(f"{method} templates accuracy on test: {weighted_acc:.2f}%")


if __name__ == "__main__":
    main()


