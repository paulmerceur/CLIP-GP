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


def build_template_text_features(model, device, classnames: List[str]) -> torch.Tensor:
    """
    Returns text features with shape [num_classes, num_templates, D].
    """
    per_class_per_template = []
    for cname in classnames:
        texts = [t.format(cname) for t in IMAGENET_TEMPLATES_SELECT]
        feats = encode_texts(model, device, texts)  # [T, D]
        per_class_per_template.append(feats)
    return torch.stack(per_class_per_template, dim=0)  # [C, T, D]


def weighted_prototypes(per_class_template_feats: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted prototypes per class.
    per_class_template_feats: [C, T, D]
    weights: [T] nonnegative, sum to 1
    returns: [C, D]
    """
    # Normalize features along D first to be safe
    feats = per_class_template_feats / per_class_template_feats.norm(dim=-1, keepdim=True)
    w = weights.view(1, -1, 1)
    proto = (feats * w).sum(dim=1)  # [C, D]
    proto = proto / proto.norm(dim=-1, keepdim=True)
    return proto


def cosine_shift(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    1 - cosine similarity per row, assuming rows are normalized.
    a, b: [C, D]
    returns: [C]
    """
    return 1.0 - (a * b).sum(dim=-1)


def evaluate(model, device, test_loader, prototypes: torch.Tensor) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["img"]
            labels = batch["label"].to(device)
            img_feats = encode_images(model, device, images)
            logits = img_feats @ prototypes.t()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / max(total, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP
    model_name = "RN50"
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # Dataset
    cfg = get_cfg_default()
    cfg.dataset.name = "Caltech101"
    env_root = os.environ.get("DATASET_ROOT")
    if env_root:
        cfg.dataset.root = env_root
    dataset = build_dataset(cfg)
    classnames = dataset.classnames

    # Loader
    test_ds = TorchDatasetWrapper(dataset.test, transform=preprocess, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    # Text features per class and template
    c_t_d = build_template_text_features(model, device, classnames)  # [C, T, D]
    C, T, D = c_t_d.shape
    base_weights = torch.ones(T, dtype=torch.float32, device=c_t_d.device) / T
    mean_prototypes = weighted_prototypes(c_t_d, base_weights)  # [C, D]

    # Evaluate baseline (uniform weights)
    base_acc = evaluate(model, device, test_loader, mean_prototypes)
    print(f"Uniform weights accuracy: {base_acc:.2f}% (T={T})")

    # Try several random weightings
    num_trials = 10
    rng = torch.Generator(device=c_t_d.device)
    rng.manual_seed(123)

    for i in range(1, num_trials + 1):
        # Dirichlet random weights (positive, sum to 1)
        alpha = torch.ones(T, device=c_t_d.device)
        # Sample gamma and normalize: equivalent to Dirichlet(alpha)
        g = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
        w = g / g.sum()

        protos_w = weighted_prototypes(c_t_d, w)

        # Prototype shift relative to uniform prototypes (mean over classes)
        shift = cosine_shift(mean_prototypes, protos_w).mean().item()

        acc = evaluate(model, device, test_loader, protos_w)
        print(f"Trial {i:02d}: acc={acc:.2f}%  shift={shift:.4f}  weights={[round(x,4) for x in w.tolist()]}")


if __name__ == "__main__":
    main()


