<h1 align="center">PurMM</h1>
<h3 align="center">
Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models
</h3>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-method">Method</a> •
  <a href="#-features">Features</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-experiments">Experiments</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 🚨 Overview

**PurMM** is a **training-free, test-time backdoor purification framework** for **Multimodal Large Language Models (MLLMs)**.

While downstream fine-tuning enables MLLMs to excel at domain-specific tasks, it also introduces severe security risks: **attackers can implant stealthy backdoors by poisoning a small portion of the fine-tuning data**. When a specific visual trigger appears, the compromised model produces attacker-chosen outputs, while behaving normally on clean inputs.

🔍 **Key idea**:  
Backdoor attacks in MLLMs exploit **Attention Hijacking** — trigger-related visual tokens dominate model attention, especially in **deep Transformer layers**, suppressing genuine visual semantics.

💡 **PurMM removes backdoors at inference time** by **locating and zeroing out anomalous visual tokens**, restoring benign and correct outputs **without retraining or modifying the model**.

<p align="center">
  <img src="PurMM_Framework.png" width="90%">
</p>

---

## 🧠 Attention Hijacking: A Hierarchical Mechanism

Through fine-grained layer-wise analysis, we uncover a **hierarchical attention pattern** in backdoored MLLMs:

- **Shallow layers**  
  Broad and normal attention distribution for basic visual feature extraction.
- **Mid-to-deep layers**  
  Attention becomes sharply concentrated on trigger-related tokens.
- **Outcome**  
  The trigger hijacks the generation process while clean performance is preserved.

This explains the paradoxical behavior of backdoored MLLMs: **high attack success rates with minimal performance degradation**.

---

## 🛠️ Method

PurMM consists of **three stages**, all executed at **test time**:

### 1️⃣ Attention-Driven Backdoor Localization

- Extract attention matrices from all Transformer layers
- Average attention across heads
- Focus on attention toward **image tokens**
- Cluster token-wise attention magnitudes
- Identify tokens with **abnormally high attention** as backdoor candidates

---

### 2️⃣ Deep-Guided Filtering Mechanism (DGFM)

Direct localization may include redundant or irrelevant tokens.  
To improve precision while preserving clean performance, PurMM leverages the **hierarchical attention mechanism**:

- Use **mid-to-deep layers** to construct a reference set
- Retain only **shallow-layer tokens** within small (3×3) neighborhoods of reference tokens
- Filter out spurious activations unrelated to the trigger

✔️ High localization accuracy  
✔️ Minimal information loss

---

### 3️⃣ Backdoor Token Zeroing-Out

- Construct a binary mask over image tokens
- **Zero out embeddings** corresponding to backdoor tokens
- Feed sanitized embeddings back into the MLLM
- Generate purified outputs

This process:
- Disrupts attention hijacking
- Preserves clean-image performance
- Restores correct answers for poisoned inputs

---

## ✨ Features

- 🧪 **Test-Time Defense**  
  Works directly on deployed MLLMs
- 🚫 **Training-Free**  
  No retraining, pruning, or data inspection
- 🎯 **Token-Level Precision**  
  Removes only trigger-related visual tokens
- 🔄 **Output Recovery**  
  Converts malicious outputs back to benign ones
- 🌍 **Model-Agnostic**  
  Validated on LLaVA and InternVL
- 🧱 **Robust to Adaptive Attacks**  
  Effective against multi-trigger and dispersed-attention strategies

---

## 🚀 Usage 

PurMM can be inserted into standard MLLM inference pipelines **between vision encoding and language generation**.

### Inference Workflow

Image + Prompt
↓
Attention Extraction (All Layers)
↓
Backdoor Token Localization
↓
Deep-Guided Filtering
↓
Zero Out Suspicious Visual Tokens
↓
Purified Generation


> ⚠️ Note: PurMM requires access to internal attention maps (white-box or gray-box setting).

---

## 📊 Experiments

### Models
- LLaVA-v1.5-7B
- InternVL2.5-8B

### Datasets
- ScienceQA (Visual Question Answering)
- IconQA (Diagram Reasoning)
- Flickr30k (Image Captioning)

### Attack Settings
- Patch / Pixel / Logo triggers
- Single-trigger and multi-trigger attacks
- Adaptive trigger dispersion

### Key Results

- 🔻 **Attack Success Rate (ASR)**: ~99% → **<5%**
- 🟢 **Clean Performance (CP)**: largely preserved
- 🔄 **Recovery Performance (RP)**: up to **83.9%**
- ⚖️ Best trade-off among all test-time baselines

PurMM consistently outperforms diffusion-based purification and token-sparsification defenses.

---

## ⚠️ Limitations & Future Work

- Slight performance drop for **semantically integrated triggers**
- Requires access to attention maps (not fully black-box)
- Future directions:
  - Black-box extensions
  - Stronger semantic-trigger defenses
  - Unified multimodal safety frameworks

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@article{jiang2026purmm,
  title   = {PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models},
  author  = {Jiang, Wenzheng and Liang, Ke and Rong, Xuankun and Zhou, Jingxuan and Zhong, Zhengyi and Wan, Guancheng and Wang, Ji},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year    = {2026}
}

