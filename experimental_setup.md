# Robust generative counterfactuals: experimental setup

*Working document for the `robust-gcf` experiments. Status: planning (November 2026).*
Items marked **[check]** still need to be verified against the method code or papers before the runs start.

---

## 1. Scope

We evaluate how robust **generative counterfactual (CF) explanation methods** are for **binary image classifiers**, in two experiments:

| Experiment | Question | What changes | Main measurements |
|---|---|---|---|
| **E1: input robustness** | Do similar inputs get similar explanations? | Small perturbations of the factual image | Distance between CFs, agreement of change maps δ = x_cf − x, sensitivity ratio, semantic (oracle) agreement, validity |
| **E2: model robustness** | Do CFs stay valid when the classifier changes? | Retrained models (seeds, data subsamples), weight-perturbed models | Validity after retraining, relaxed stability, invalidation rate |

**Direction of all CFs:** positive class → negative class, for every method and task (section 4).

---

## 2. Data


| | **CelebA** | **ChestMNIST+** (NIH ChestX-ray14 via MedMNIST+) |
|---|---|---|
| Role | Natural images, two tasks | Medical images, one task |
| Size | 202,599 images; official split 162,770 / 19,867 / 19,962 | ~112k images; official MedMNIST splits |
Source classes | Smiling, Young | Effusion
Target classes | Not smiling, Old | Healthy

All methods generate CFs **from the positive to the negative class only**:
Smiling → Not smiling, Young → Old, Pneumonia → Healthy.

### Evaluation set (shared by all CF methods)

- **100 images per task**, taken from the test split, all **predicted positive by M0 with confidence ≥ 0.7** (and correctly labeled).
- Fixed once, before any CF is generated.
- **Perturbations (E1), seeded and shared across methods:** Gaussian noise σ (pixels in [0,1]); (optionally, one natural perturbation: pixel shift or ±5% brightness). Keeping a perturbed image only if M0's prediction stays the same and |p(x) − p(x′)| < 0.05.

---

## 3. Classification tasks

CelebA:  Smiling -> Not smiling (≈ balanced sample)

CelebA:  Young -> Old (imbalanced: ≈ 25% Not Young)

ChestMNIST: Effusion -> Healthy (imbalanced: ≈ 18% Effusion)


### Architecture and interface

| Setting | Value |
|---|---|
| Architecture | **DenseNet121**  (matches the classifiers DiME and GCD were built with) |
| Input | Resize to 128 (bilinear, antialias) → normalize (ImageNet mean/std for RGB; mean 0.5 / std 0.5 for grayscale, configurable) |
| Output | **One logit** for the positive class; probabilities via sigmoid; decision threshold stored with the model |


### Training defaults

- ImageNet-pretrained initialization, mixed precision, ~5 epochs (CelebA) / ~10 epochs (ChestMNIST), early stopping on validation AUROC.
- Class imbalance (Young, Effusion): `pos_weight` in the loss **or** a threshold chosen on the validation set; the choice is fixed per task and recorded.
- Augmentation: horizontal flip for CelebA only; none for ChestMNIST.

---

## 6. Native format of each generative model

| Method | Generative model | CelebA input | ChestMNIST input |
|---|---|---|---|
| REVISE | VAE (ours) | 128, RGB | 128, gray | 
| DiME | DDPM (guided-diffusion UNet) | 128, RGB (pretrained) | 128, gray (would need training) | 
| GCD | Diffusion autoencoder (DiffAE) | 256, RGB (CelebA-HQ trained) | would need training from scratch | 
| COIN | Conditional GAN (encoder–generator with skips, SNGAN-style discriminator) | 128, RGB | 128, gray |
| HAVE | Hierarchical conditional VAE | 128, RGB | 128, gray |

Classifier input is always 128, whatever the generator's resolution.

---

## 7. Counterfactual methods

### 7.1 Overview and taxonomy

| Method | Paper | Taxonomy path | Classifier access | Coupled? | Code | Framework |
|---|---|---|---|---|---|---|
| **COIN** | Shvetsov et al. 2024 | Feature · Amortized gen. · Coupled · White-box | White-box (gradients through the frozen classifier) | Yes | github.com/Dmytro-Shvetsov/counterfactual-search | PyTorch 2.0 + Lightning Fabric |
| **GCD** | Sobieski & Biecek 2024 | Latent · Direct latent manip. · Inherited · Unsupervised · Post-hoc | Black-box (proxy model) | No | github.com/sobieskibj/gcd | PyTorch |
| **REVISE** | Joshi et al. 2019 | Latent · Instance-level opt. · Unstructured · Unsupervised · Post-hoc | White-box | No | Re-implemented in PyTorch (original: TensorFlow) | PyTorch |
| **DiME** | Jeanneret et al. 2022 | Feature · Guided denoising · Unstructured · Unsupervised · Post-hoc | White-box | No | github.com/guillaumejs2403/DiME | PyTorch |
| **HAVE** | Vercheval & Pižurica 2021 | Latent · Direct latent manip. · Enforced · Unsupervised · Coupled | Black-box (classifier probabilities as training signal)| Yes | github.com/nverchev/VAEX | DryTorch (PyTorch-based) |


### 7.2 Generative components

| Component | COIN | GCD | REVISE | DiME | HAVE |
|---|---|---|---|---|---|
| Backbone | Encoder + conditional generator (skip connections), spectral-norm discriminator | DiffAE: semantic encoder + conditional DDIM decoder | VAE (encoder + decoder) | Unconditional DDPM | Hierarchical conditional VAE |
| Edit mechanism | Generator outputs a **residual** added to the input (`ptb_based: True`), conditioned on the classifier's prediction bin | Move the semantic latent along a **global direction** by a chosen step size | **Gradient optimization** of the latent per image | Noise to step τ, then **classifier-guided denoising** | Change the class-encoding latent |
| Getting the image in | Encoder (feed-forward) | Semantic encoder + deterministic DDIM inversion | Encoder mean as the start point | Forward noising (stochastic) | Hierarchical encoder |
| Classifier role | KL loss f(x_cf) ≈ desired probability during training | Proxy trained on M0's outputs; directions found with the proxy | Loss at every optimization step | Gradient on the denoised estimate at every step | Probabilities used in training |
| Randomness | Training seed only; deterministic at inference | Proxy training and direction search; inversion deterministic | Deterministic (encoder mean) | **High** (DDPM sampling): needs a two-seed baseline in E1 | Low with latent means |
| Hyperparameters to fix | `lambda_adv 1.0`, `lambda_kl 5.0`, `lambda_rec 1.0`, `lambda_tv 0.05`, `cyclic_rec: True`, `num_bins 2`; training steps | Number of directions, step size, proxy architecture, DDIM steps | λ (proximity), learning rate, max steps, stopping confidence | τ = 60 of 200 respaced steps, λc ∈ {8, 10, 15}, λp = 30, λ1 = 0.05 | Latent level, size of the change, KL weights |
| Extra outputs to store | Difference map \|x − x_cf\| | Semantic latent, inversion noise, direction, step | Initial/final latent, steps | Successful λc, success flag | Latents per level |

### 7.3 Adaptation per dataset

| Method | CelebA (Smiling, Young) | ChestMNIST |  
|---|---|---|
| **COIN** | Add a CelebA dataset class returning `{'image','label','masks'}` (dummy masks; v2 does not use them); `in_channels: 3`; `img_size: [128,128]`; plug in DenseNet121; cap training by steps (configs use 151 epochs on CT) | Same with `in_channels: 1` (pathology → normal) | 
| **GCD** | DiffAE pretrained on CelebA; proxy + directions per target provided | **DiffAE must be trained from scratch** |
| **REVISE** | Train one VAE on CelebA (shared by both targets) | Train one VAE on ChestMNIST | 
| **DiME** | Pretrained CelebA DDPM, classifier and oracle from the DiME repo; swap in our M0 | **No checkpoint known** | 
| **HAVE** | Train with M0 per target (original used CelebA Female/Male with its own 5-stage CNN) | Train with M0-Effusion | 


---

## 8. Repository structure (`robust-gcf`)

One repository; one environment per CF method; methods communicate with the evaluation **only through files** in `outputs/`.

```
robust-gcf/
├── README.md
├── configs/                    # Hydra
│   ├── config.yaml
│   ├── dataset/   celeba.yaml, chestmnist.yaml
│   ├── target/    celeba_smiling.yaml, celeba_young.yaml, chestmnist_pneumonia.yaml
│   ├── model/     densenet121.yaml
│   ├── train/     default.yaml
│   ├── variant/   m0.yaml, seed.yaml, weight_noise.yaml
│   ├── method/    coin.yaml, gcd.yaml, revise.yaml, dime.yaml, have.yaml
│   └── exp/       input_robustness.yaml, model_robustness.yaml
|
├── src/robust_gcf/
│   ├── core/                   # NO torch/tf imports; installed into every method env
│   │   └── schemas.py          # pydantic: TargetSpec, ClassifierMeta, manifests, CF output schema
│   │   
│   ├── data/
│   │   ├── sources/   celeba.py, chestmnist.py
│   │   ├── targets.py          # attribute/finding → binary label policies
│   │   ├── preprocessing.py    # transform, train augmentation
│   │   ├── cache.py            # uint8 memmap cache + index.parquet
│   │   ├── datamodule.py       # LightningDataModule (subsample, class balance)
│   │   ├── evalset.py          # evaluation-set selection
│   │   └── perturbations.py    # seeded perturbations
|   |
│   ├── models/
│   │   ├── factory.py          # build_classifier(cfg)
│   │   ├── wrapper.py          # ClassifierWrapper: [0,1] → resize → normalize → logit
│   │   ├── checkpoint.py       # save/load with embedded config
│   │   ├── export.py           # ONNX export + parity check
│   │   └── variants.py         # weight noise, randomized
|   | 
│   ├── training/               # LitClassifier, train entry point, threshold calibration
│   ├── evaluation/             # metrics/, robustx_adapter.py, exp1, exp2, stats
│   └── utils/
|
├── cf_methods/                    # one folder per CF method, each with its own environment
│   ├── revise/   wrapper.py, env.yml
│   ├── dime/     upstream/ (submodule), wrapper.py, env.yml, patches/
│   ├── gcd/      upstream/, wrapper.py, env.yml, patches/
│   ├── coin/     upstream/ (counterfactual-search), wrapper.py, env.yml, patches/
│   ├── have/     upstream/ (VAEX), wrapper.py, env.yml, patches/
│   
├── scripts/                    # CLI entry points
├── notebooks/
├── data/                       # gitignored: raw/, cache/
└── outputs/                    # gitignored
    ├── classifiers/{dataset}/{target}/{variant}/   model.pt, config.yaml, metrics.json
    ├── evalsets/{dataset}/{target}/                manifest.parquet, images.zarr, perturbations.zarr
    ├── cfs/{method}/{dataset}/{target}/            cfs.zarr, meta.parquet
    └── metrics/                                    *.parquet
```

---

## 9. ToDos before the runs

1. CelebA crop: match DiME's preprocessing exactly.
2. GCD: availability and resolution of the authors' CelebA DiffAE; reconstruction quality on our CelebA crops.
3. DiME / GCD on ChestMNIST: check usable checkpoints (FastDiME_Med).
4. HAVE: input resolution (128 vs 129) and how classifier outputs enter training.
