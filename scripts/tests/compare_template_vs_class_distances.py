import torch
import itertools
from typing import List

# Local imports from this repo
from clip import clip
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT


def encode_templates_for_classes(model, device: torch.device, classnames: List[str]) -> List[torch.Tensor]:
    """
    For each classname, encode text prompts for all templates and return a list of
    tensors with shape [num_templates, embed_dim]. Embeddings are L2-normalized.
    """
    template_texts_per_class: List[List[str]] = [
        [template.format(classname) for template in IMAGENET_TEMPLATES_SELECT]
        for classname in classnames
    ]

    with torch.no_grad():
        per_class_embeddings: List[torch.Tensor] = []
        for texts in template_texts_per_class:
            tokens = clip.tokenize(texts).to(device)
            emb = model.encode_text(tokens)
            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            per_class_embeddings.append(emb)

    return per_class_embeddings


def cosine_distance_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distance matrix for L2-normalized embeddings.
    embeddings: [N, D] L2-normalized
    returns: [N, N] with distances in [0, 2]
    """
    # For normalized vectors, cosine similarity is dot product
    sims = embeddings @ embeddings.t()
    return 1.0 - sims


def summarize(values: torch.Tensor) -> str:
    values = values.flatten()
    return (
        f"mean={values.mean().item():.4f} "
        f"std={values.std(unbiased=False).item():.4f} "
        f"min={values.min().item():.4f} "
        f"max={values.max().item():.4f} "
        f"n={values.numel()}"
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classnames = [
        "goldfish",
        "tiger cat",
        "airliner",
        "volcano",
        "espresso",
    ]

    print(f"Using device: {device}")
    print(f"Num templates: {len(IMAGENET_TEMPLATES_SELECT)}")
    print("Suggested classes:")
    for c in classnames:
        print(f"  - {c}")

    # Ask user for the main class
    main_class = input("\nChoose a main class: ").strip()
    if not main_class: main_class = classnames[0]
    other_classes = [c for c in classnames if c != main_class]
    
    print(f"Main class: {main_class}")
    print(f"Comparing against {len(other_classes)} classes.")

    # Load CLIP
    model_name = "RN50"
    model, _ = clip.load(model_name, device=device, jit=False)
    model.eval()

    # Encode template embeddings for main class and others
    per_class_embeddings = encode_templates_for_classes(
        model, device, [main_class] + other_classes
    )

    # Intra-class distances for the main class only (per-template to class mean)
    main_emb = per_class_embeddings[0]

    # Inter-class distances (per-template): distance of each other class's template
    # embedding to the main class mean. This reduces template-induced variance when
    # comparing classes and aligns the scale with the per-template intra stats.
    class_means = []
    for emb in per_class_embeddings:
        mean_emb = emb.mean(dim=0, keepdim=True)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)
        class_means.append(mean_emb)
    class_means = torch.cat(class_means, dim=0)  # [1 + K, D]

    main_mean = class_means[0:1]  # [1, D]
    # Per-template distances to main class mean become the intra-class distances we summarize
    per_template_sims = (main_emb @ main_mean.t()).squeeze(1)
    per_template_dists = (1.0 - per_template_sims)
    intra_dists_main = per_template_dists
    per_class_template_dists = []  # list of [M] tensors
    per_class_mean_dists = []      # list of floats (mean per class)
    for emb in per_class_embeddings[1:]:  # each is [M, D]
        sims_t = (emb @ main_mean.t()).squeeze(1)  # [M]
        dists_t = (1.0 - sims_t)                   # [M]
        per_class_template_dists.append(dists_t)
        per_class_mean_dists.append(dists_t.mean().item())

    inter_dists = torch.cat(per_class_template_dists, dim=0)

    # Print summaries
    print("\nIntra-class distances across templates for main class (cosine distance):")
    print("  "+summarize(intra_dists_main))
    print("\nPer-template distances to main class mean:")
    for tmpl, dist in zip(IMAGENET_TEMPLATES_SELECT, per_template_dists.tolist()):
        print(f"  {tmpl.format(main_class)}: {dist:.4f}")

    print("\nInter-class distances (per-template to main class mean) (cosine distance):")
    print("  "+summarize(inter_dists))
    print("\nPer-class mean distances (averaged over that class's templates):")
    for cname, dist in zip(other_classes, per_class_mean_dists):
        print(f"  {cname}: {dist:.4f}")


if __name__ == "__main__":
    main()


