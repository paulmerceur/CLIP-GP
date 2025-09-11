import os
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP
    model_name = "RN50"
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # Build Caltech101 dataset (uses default root from config.py; override via env ROOT if needed)
    cfg = get_cfg_default()
    cfg.dataset.name = "Caltech101"
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        cfg.dataset.root = env_root

    dataset = build_dataset(cfg)
    classnames = dataset.classnames

    # Prepare test loader
    test_ds = TorchDatasetWrapper(dataset.test, transform=preprocess, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
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

    # Evaluate per-template
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
        print(f"{tmpl}: {acc:.2f}%")


if __name__ == "__main__":
    main()


