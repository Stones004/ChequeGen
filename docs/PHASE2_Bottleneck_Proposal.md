# Phase 2 — Bottleneck Selection & Proposal
### Project: *ChequeGen — Fast Diffusion-Based Synthetic Handwritten Cheque Amounts for Banking OCR*

---

## 1. Selection

| | |
|---|---|
| **Architecture** | Diffusion Model (DDPM) |
| **Limitation** | **Slow generation speed.** Sampling needs T = 1000 sequential denoising steps. |
| **Impact on use-case** | Generating large synthetic cheque datasets for OCR training becomes slow and costly. |

---

## 2. Proposed Modification

Two changes are proposed: **DDIM deterministic sampling** (a mathematical modification), plus **class-conditioning with classifier-free guidance** (a structural modification) so that specific digits can be generated.

### 2.1 Mathematical modification: DDIM sampling
DDPM's reverse process is a **Markov chain**, so every one of the 1000 steps must be run. DDIM (Song et al., 2021) redefines it as a **non-Markovian** process. The model can then **skip steps**, jumping along a short sub-sequence (e.g. 50 steps) of the original 1000:

$$
x_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}\right)}_{\text{predicted } x_0} + \sqrt{1-\bar\alpha_{t-1}}\;\epsilon_\theta(x_t,t)
$$

- **The same trained network is used**, with no retraining. Only the sampling equation changes.
- Steps are reduced from 1000 to **50 / 20 / 10**, so generation should be roughly **20–100× faster**.

### 2.2 Structural modification: class-conditional U-Net
- The digit label (0–9) is added as an embedding to the time embedding inside the U-Net.
- **Classifier-free guidance:** the label is randomly dropped 10% of the time during training. At sampling:
  ε̂ = (1 + w)·ε_θ(x_t, t, y) − w·ε_θ(x_t, t, ∅)
- This lets us generate **any amount on demand**, e.g. "45750" = digits 4, 5, 7, 5, 0.

---

## 3. Dataset

| Item | Detail |
|---|---|
| **Dataset** | **MNIST** (70,000 handwritten digits, 28×28, labels 0–9); EMNIST optional for extra styles |
| **Why** | A public, privacy-free stand-in for handwritten cheque digits, small enough to train on free Colab GPU |
| **Pre-processing** | Normalise to [−1, 1]; generated digits are later stitched into an amount strip and placed on a cheque template |

---

## 4. Benchmarking Plan (Phase 3)

| Model | Description |
|---|---|
| **Baseline** | Conditional DDPM, 1000-step sampling |
| **Improved** | Same model + DDIM sampling at 50, 20 and 10 steps |

| Metric | Purpose |
|---|---|
| **Inference latency** (sec per 1000 images) | Proves the speed-up (main metric) |
| **FID score** | Shows quality is maintained |
| **OCR accuracy** | Trains a small CNN classifier on synthetic digits and tests on real MNIST, to check the synthetic data is actually useful |

**Expected result:** about 20× faster at 50 steps with similar FID. A speed/quality trade-off curve will be plotted for 10–1000 steps.

---

## 5. Real-World Deployment (Phase 4)

**Gradio app: "ChequeGen"**
1. The user types an amount (e.g. `45750`).
2. The model generates each digit in a random handwriting style using DDIM (fast).
3. The digits are placed into the **amount box of a sample cheque template**, labelled "SPECIMEN".
4. Options: number of styles, sampling steps (slider showing the speed vs quality trade-off), and a button to **download a synthetic dataset** (images + labels) for OCR training.

**Users:** bank/fintech teams building cheque OCR, and researchers who need privacy-safe handwriting data.

---

## 6. Timeline

| Dates | Task |
|---|---|
| 24–25 Nov | Train conditional DDPM on MNIST (Colab, ~30–40 min) |
| 26 Nov | Implement DDIM sampler, measure latency + FID |
| 27 Nov | OCR-usefulness test, plots |
| 28 Nov | Gradio app with cheque template |
| 29 Nov | GitHub README + submission |

---

## References
1. Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.
2. Song, Meng & Ermon (2021). *Denoising Diffusion Implicit Models.* ICLR.
3. Ho & Salimans (2022). *Classifier-Free Diffusion Guidance.*
4. LeCun et al. (1998). *MNIST handwritten digit database.*
5. Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule (FID).*
