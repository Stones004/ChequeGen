# Phase 1 — Comparative Study of Deep Generative Architectures
### Project: *ChequeGen — Fast Diffusion-Based Synthetic Handwritten Cheque Amounts for Banking OCR*

**Course:** CAC — Generative Architectures | **Phase:** 1 (Report) | **Duration:** 13 Nov – 29 Nov 2026

---

## 1. Motivation

Indian banks still clear large volumes of paper cheques through the **Cheque Truncation System (CTS)**. Each cheque image is scanned, and the **handwritten amount** has to be read and checked against the amount written in words. Training OCR models for this needs large amounts of labelled handwritten digit data. Real cheques **cannot be shared** because of customer privacy and banking regulation.

**Generative models can produce unlimited, privacy-free synthetic handwritten amounts.** This project studies four generative architectures and selects one to build a practical, fast generator.

---

## 2. Working of Each Architecture

### 2.1 Transformer
- Self-attention: Attention(Q,K,V) = softmax(QKᵀ/√d)V.
- Generates autoregressively, one token (or image patch) at a time: p(x) = ∏ p(xᵢ | x<ᵢ).
- Trained with next-token prediction (cross-entropy).

### 2.2 GAN
- Generator G(z) creates images from noise; discriminator D(x) classifies real vs fake.
- Minimax objective: min_G max_D 𝔼[log D(x)] + 𝔼[log(1 − D(G(z)))].
- Sampling takes one forward pass.

### 2.3 VAE
- Encoder q(z|x) → latent Gaussian; decoder p(x|z) reconstructs.
- Maximises the ELBO: 𝔼[log p(x|z)] − KL(q(z|x) ‖ 𝒩(0, I)).
- Sampling takes one forward pass.

### 2.4 Diffusion Model (DDPM)
- **Forward process:** gradually adds Gaussian noise over T = 1000 steps, x_t = √ᾱ_t x₀ + √(1−ᾱ_t) ε.
- **Reverse process:** a U-Net ε_θ(x_t, t) learns to predict the noise, trained with the simple loss ‖ε − ε_θ(x_t, t)‖².
- **Sampling:** start from pure noise and denoise step by step, which takes **1000 network passes**.

---

## 3. Comparative Analysis

| Criterion | Transformer | GAN | VAE | Diffusion |
|---|---|---|---|---|
| **Training stability** | High, but data-hungry | **Low** (mode collapse, oscillation) | High | **High** (simple MSE loss) |
| **Output quality** | Good for text; average for images | Sharp, but low diversity | **Blurry** | **Best** (sharp + diverse) |
| **Generation speed** | Slow (sequential) | **Fast** (1 pass) | **Fast** (1 pass) | **Very slow** (1000 passes) |
| **Latent interpretability** | Low | Low (entangled) | **High** (smooth latent) | Low (latent = noise image) |
| **Mode coverage** | Good | Poor | Good | **Excellent** |

### Advantages and Disadvantages

| Architecture | Advantages | Disadvantages |
|---|---|---|
| **Transformer** | Long-range dependencies, exact likelihood, scalable | Quadratic attention cost, needs huge data, slow sequential generation |
| **GAN** | Sharp images, fast sampling | Mode collapse (e.g. only generates a few digit styles), unstable training, no likelihood |
| **VAE** | Stable training, fast, interpretable latent | Blurry outputs, so digit strokes look smudged; poor for OCR training data |
| **Diffusion** | Highest quality and diversity, stable training | **Slow sampling** (1000 steps), high inference cost |

---

## 4. Why This Matters for Cheque OCR

| Requirement | Best fit |
|---|---|
| Realistic, sharp strokes (OCR needs crisp digits) | Diffusion |
| Many handwriting styles (no mode collapse) | Diffusion |
| Generating **lakhs of training samples** quickly | GAN / VAE |
| Stable training on a student GPU (Colab) | Diffusion / VAE |

**Conclusion:** Diffusion gives the best quality and diversity, which are exactly what OCR training data needs. Its one major weakness is **slow generation**: making 1,00,000 synthetic cheque amounts with 1000-step sampling is very expensive. Phase 2 selects the **Diffusion model** and targets this bottleneck.

---

## References
1. Vaswani et al. (2017). *Attention Is All You Need.*
2. Goodfellow et al. (2014). *Generative Adversarial Nets.*
3. Kingma & Welling (2014). *Auto-Encoding Variational Bayes.*
4. Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models.*
