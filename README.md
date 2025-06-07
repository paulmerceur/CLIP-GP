# CLIP‑GP

> **Gaussian‑Process Weighted Template Adaptation for CLIP**  

> Paul Merceur ⋅ ÉTS Montréal

CLIP‑GP extends the original **CLAP** repository with a *Gaussian‑Process (GP)*
head that learns to **weight template embeddings** during few‑shot adaptation.
Everything else—dataset loaders, training script, CLI—remains fully compatible
with the upstream codebase.

---

## Highlights

* **Drop‑in replacement** for ZS‑LP / CLAP: enable GP weighting with a single
  flag (`--use_gp`).
* **Fast** cached text embeddings + batched variational inference keep the
  overhead < 5 % w\.r.t. vanilla linear probing.
* **Library first** core GP logic is implemented with
  [GPyTorch](https://gpytorch.ai/) for robustness and readability.

---

## Installation

1. **Dassl + PyTorch** – follow the
   [Dassl.pytorch installation guide](https://github.com/KaiyangZhou/Dassl.pytorch#installation).
2. Activate the `dassl` conda env and run:

   ```bash
   pip install -r requirements.txt
   ```

   This installs the extra packages needed by CLIP and GPyTorch.
3. Download the datasets listed in `DATASETS.md` (identical to CLAP).

<details>
<summary>GPU &amp; mixed‑precision notes</summary>

* FP16/AMP are fully supported. 
* Multi‑GPU training is unchanged—GP parameters live on the same device as the
  adapter.

</details>

---

## Usage

We reuse the original driver scripts—just pass an extra argument to enable the
GP head.

### (a) Zero‑shot‑initialised Linear Probe *(ZS‑LP)*

```bash
bash scripts/adapt.sh 0 imagenet SGD_lr1e-1_B256_ep300 1 ZS none RN50
```

### (b) CLass‑adaptive Linear Probing *(CLAP)*

```bash
bash scripts/adapt.sh 0 imagenet SGD_lr1e-1_B256_ep300 1 ZS l2 RN50
```

### (c) **CLIP‑GP** (this project)

```bash
bash scripts/adapt.sh 0 imagenet SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 \
    --use_gp true \
    --gp_beta 0.3 --gp_lengthscale 1.0 --gp_num_mc_samples 5
```

Key new flags (all have sensible defaults):

| Flag                                                 | Description                            |
| ---------------------------------------------------- | -------------------------------------- |
| `--use_gp`                                           | Enable the GP weighting module.        |
| `--gp_beta`                                          | KL term weight in the ELBO.            |
| `--gp_lengthscale`, `--gp_outputscale`, `--gp_noise` | Kernel hyper‑parameters.               |
| `--gp_num_mc_samples`                                | # MC samples for prototype estimation. |

### (d) Domain generalisation test

```bash
bash scripts/eval.sh 0 imagenet imagenetv2 SGD_lr1e-1_B256_ep300 1 ZS l2 RN50 --use_gp true
```

---

## Repository layout

```
CLIP-GP/
├── configs/                    # config files for easier setups
├── datasets/                   # dataset helpers & templates
├── gp_template_weighter.py     # new – GP module (GPyTorch)
├── trainers/
│   └── adapters.py             # updated adapter with caching & GP support
├── scripts/                    # same driver scripts + new GP flags
└── README.md                   # you are here
```

---

## Acknowledgements

CLIP‑GP builds on the excellent **CoOp**, **CLAP**, and **TaskRes** codebases.
Huge thanks to their authors for open‑sourcing the groundwork.

---

## License

This repository inherits the MIT license of the original CLAP project.

---

## Contact

Questions or issues? Open a discussion or contact me at `paul.merceur.1@ens.etsmtl.ca`.
