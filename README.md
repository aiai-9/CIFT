# CIFT: Donor-Grounded Identity Gap Learning for Source-Free Face Forgery Detection

---

## Overview

**CIFT** (**C**onsistency-**I**nvariant **F**orensic **T**raining) is a privileged-learning framework for source-free face-swap forgery detection.

Most deepfake detectors rely on renderer-specific artifact cues (frequency anomalies, blending boundaries) that shift across synthesis pipelines and degrade on unseen forgeries. CIFT exploits a more stable signal: every face-swap grafts a *donor* identity onto a *recipient* face, imposing a cross-person identity mismatch that is intrinsic to the swap operation and pipeline-invariant.

During training, **actual donor reference frames** from FaceForensics++ are used as privileged inputs to supervise an explicit cross-person identity gap through two training-only components:

- **XID-Mamba** (Cross-Identity Mamba) — encodes donor and target streams with independent bidirectional Mamba scans and symmetric cross-attention to compute the gap Δ = ‖g_s − g_t‖₂.
- **Type-Conditioned IGS** (Identity Gap Supervision) — imposes cosine and Euclidean geometric separation with forgery-type-dependent margins (swap vs. reenactment).

At inference, **all donor-dependent modules are discarded**. The deployed model is fully source-free: it classifies a single observed face using only the Global Head and Spatial Mamba Classifier (SMC), with zero privileged overhead.

Trained on FF++ (c23) and evaluated zero-shot across five unseen benchmarks, CIFT achieves **92.61 average AUC**, outperforming the strongest baseline (DiffusionFake) by **+10.73 AUC**, with the largest gains on WildDeepfake (+16.81) and DiffSwap (+14.45) where diffusion rendering suppresses conventional artifact cues.

---

## Architecture

![CIFT Architecture](figures/cift_7.png)

During training, the target face *x* and the privileged donor *x_s* share backbone *E*, then diverge into independent upsampling and FeatureFilter branches. Post-filter features condition a frozen SD 1.5 auxiliary pathway and feed XID-Mamba, which computes the cross-person gap Δ supervised by IGS. At inference all training-only branches (marked †) are discarded; only the Global Head and SMC classify the observed face.

```
Training:   x, x_s ──► Backbone E ──► Global Head + SMC + XID-Mamba† + Diffusion† + IGS†
Inference:  x       ──► Backbone E ──► Global Head + SMC  →  ŷ_sf
```

`†` = training-only, discarded at deployment.

---

## GradCAM diagnostics

![GradCAM activations](figures/gradcam_cift_dfake.png)

On genuine faces both models produce diffuse, low-confidence activations (CIFT ≤ 0.13). On forged faces CIFT concentrates sharply on the periocular zone, nasal bridge, and jawline — the anatomical loci of donor–recipient mismatch — at 0.94–0.96 confidence across GAN-based (Celeb-DF) and diffusion-based (DiffSwap) forgeries. DiffusionFake activations remain spatially diffuse and its confidence drops to 0.70–0.80 on DiffSwap.

---

## Environment setup

**Requirements:** Python 3.9+, PyTorch ≥ 2.0, CUDA 11.8+

```bash
# Clone the repository
git clone https://github.com/aiai-9/CIFT.git
cd CIFT

# Create and activate a conda environment
conda create -n cift python=3.9 -y
conda activate cift

# Install PyTorch (adjust cuda version as needed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Pre-trained model preparation

CIFT uses a frozen Stable Diffusion 1.5 prior as an auxiliary training-time semantic regulariser.

**Step 1.** Download the SD 1.5 checkpoint from Hugging Face:
```bash
# Download v1-5-pruned.ckpt from https://huggingface.co/runwayml/stable-diffusion-v1-5
mkdir -p models/
# Place the downloaded file at: models/v1-5-pruned.ckpt
```

**Step 2.** Convert to ControlNet-compatible format:
```bash
python tool_add_control.py models/v1-5-pruned.ckpt models/control_sd15_ini.ckpt
```

This generates `models/control_sd15_ini.ckpt`, which CIFT uses as the frozen SD 1.5 backbone. The SD U-Net, VAE, and CLIP encoder receive **no gradient updates** during CIFT training.

---

## Dataset preparation

### FaceForensics++ (FF++) — training

All CIFT models are trained exclusively on FF++ at compression level **c23**.

Organise the dataset in the following structure:

```
/path/to/ffpp/
├── manipulated_sequences/
│   ├── Deepfakes/
│   │   └── c23/
│   │       ├── 000_003/
│   │       │   ├── 000_003_0000.png
│   │       │   └── ...
│   │       └── ...
│   ├── Face2Face/
│   │   └── c23/
│   ├── FaceSwap/
│   │   └── c23/
│   └── NeuralTextures/
│       └── c23/
└── original_sequences/
    └── youtube/
        └── c23/
            ├── 000/
            │   ├── 000_0000.png
            │   └── ...
            └── ...
```

FF++ also provides **source videos** for each forged sequence. CIFT uses these to retrieve the temporally nearest source-identity frame as the privileged donor reference `x_s` for each forged sample. For genuine frames, `x_s = x` (yielding Δ = 0 by construction). The relation-aware data loader handles donor retrieval automatically; set `data_root` in the config to your FF++ root.

### Evaluation benchmarks

Five held-out benchmarks are evaluated zero-shot with no fine-tuning:

- **Celeb-DF v2** — high-quality GAN-based celebrity swaps ([GitHub](https://github.com/yuezunli/celeb-deepfakeforensics))
- **WildDeepfake** — in-the-wild internet forgeries ([GitHub](https://github.com/deepfakeinthewild/deepfake-in-the-wild))
- **DFDC-Preview** — diverse acquisition conditions ([Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge/data))
- **DFD** — studio-quality source-separated recordings ([Google AI](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html))
- **DiffSwap** — diffusion-based face swaps, 6,000-image partition ([GitHub](https://github.com/wl-zhao/DiffSwap))

No donor reference, source-video identifier, or pairing metadata is available or used at any test benchmark. All predictions are produced by the source-free two-branch inference path. Update the `data_root` path in `configs/cift/test.yaml` before running evaluation.

---

## Training

### Default protocol

Train CIFT on FF++ c23 with the ConvNeXt-V2-Base backbone:

```bash
# Single-node, 4× A100 (bf16 DDP, ~12 h wall-clock)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    -c configs/cift/train_cift_generalized.yaml
```

Key hyperparameters (backbone: ConvNeXt-V2-Base, resolution: 256×256, optimiser: AdamW weight-decay 5e-4, base LR: 2e-5 with 1.2× head multiplier, scheduler: OneCycleLR 8% warmup + cosine, effective batch: 128, epochs: 80, bf16 DDP, gradient clip norm 0.5, XID-Mamba dual-branch dropout: 0.15). Full details are in `configs/cift/train_cift_generalized.yaml`.

### Multi-domain fine-tuning

After FF++-only pretraining, fine-tune on the mixed five-domain set:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    -c configs/cift/train_cift_generalized.yaml \
    dataset.name=mixed \
    train.lr=1e-5 \
    train.epochs=60 \
    train.resume_ckpt=auto
```

---

## Evaluation

Run zero-shot cross-dataset evaluation with a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    -c configs/cift/test.yaml \
    --ckpt experiments/CIFT_ffpp_only/ckpt/cift-best-eer-epoch=XX.ckpt \
    --dataset celeb_df        # options: celeb_df | wild_deepfake | dfdc_p | dfd | diffswap
```

Frame-level AUC and EER are reported by default. For video-level evaluation (mean-pooled frame logits):

```bash
python test.py ... --video_level
```

---

## Repository structure

```
CIFT/
├── configs/
│   └── cift/
│       ├── train_cift_generalized.yaml   # main training config
│       └── test.yaml                     # evaluation config
├── cldm/
│   ├── mamba_modules.py    # XID-Mamba, SMC, IGS, MambaFakeHead
│   └── model.py            # full CIFT model definition
├── datasets/
│   ├── base_dataset.py
│   ├── mixed_dataset.py    # multi-domain weighted sampler
│   ├── forgery_type_mapper.py  # FF++ family → type token routing
│   └── data_structure.py
├── analysis/
│   ├── gap_orthogonality.py    # analysis-time Δ diagnostic
│   └── gradcam_visualise.py    # GradCAM activation maps
├── figures/
│   ├── cift_7.png              # architecture overview
│   └── gradcam_cift_dfake.png  # GradCAM comparison
├── models/                     # SD 1.5 checkpoint directory (not tracked)
├── experiments/                # training outputs (not tracked)
├── train.py
├── test.py
├── tool_add_control.py         # SD → ControlNet conversion
└── requirements.txt
```

---

## Forgery-type routing

XID-Mamba uses two learnable type tokens to condition the bilateral Mamba scan on the manipulation regime. Deepfakes and FaceSwap map to `τ_swap` (cosine push margin 1.0, Euclidean margin 1.2); Face2Face and NeuralTextures map to `τ_reenact` (margins 0.8 / 0.9); genuine frames use no token and a pull tolerance δ = 0.3. Non-FF++ datasets default to `swap` routing at training time. Type-token mapping is defined in `datasets/forgery_type_mapper.py`.

---

## Loss weights

All weights selected by grid search on the FF++ validation split: λ_cls = 1.5 (cross-entropy + focal), λ_diff = 0.5 (diffusion denoising), λ_ig = 0.15 (IGS cosine contrastive), λ_gap = 0.1 (IGS Euclidean regression), λ_a = 1e-4 (cross-attention regularisation), λ_β = 0.05 (fusion temperature regularisation).

---

## Deployed model complexity

At inference, only the backbone, projection head, Global Head, and SMC are retained (103.4 M parameters for ConvNeXt-V2-B). All training-only modules — XID-Mamba, IGS, type tokens, donor upsampling branch, diffusion pathway — are discarded with zero privileged inference overhead.

---

## Reproducibility

Three-seed stability (seeds varied, data splits and hyperparameters fixed): CIFT achieves 95.28 ± 0.16 AUC on Celeb-DF and 96.46 ± 0.18 AUC on DiffSwap. The maximum standard deviation (0.18) is well below the reported margins over all baselines.

---

## Ethical considerations

CIFT is intended for defensive forensic analysis and deepfake misuse mitigation. It is **not** intended for identity attribution, surveillance, or automated punitive decision-making. Donor references are used only during training on licensed benchmark data and are never required at deployment. Model outputs should be treated as decision-support evidence rather than final identity judgements. Deployment should report calibrated uncertainty across demographic and acquisition-condition subgroups.

---

## Citation

```bibtex
@article{cift2025,
  title  = {CIFT: Donor-Grounded Identity Gap Learning for Source-Free Face Forgery Detection},
  author = {Anonymous},
  year   = {2025},
}
```

---

## License

The codebase builds on components with the following licenses:

- Stable Diffusion 1.5 — CreativeML Open RAIL-M
- ControlNet — Apache 2.0
- ConvNeXt-V2 — MIT
- Mamba — Apache 2.0
