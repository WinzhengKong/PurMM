<h1 align="center">PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models


## Overview

This repository is based on the paper **PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models**.

The paper studies a practical security problem in **multimodal large language models (MLLMs)**: when a general-purpose MLLM is fine-tuned on downstream data, an attacker can poison a small portion of the training set and implant a **backdoor**. Once a trigger appears in the input image, the fine-tuned model may ignore normal semantics and produce an attacker-chosen target output.

Instead of retraining the model or modifying the fine-tuning pipeline, PurMM proposes a **test-time defense**. The idea is to analyze attention patterns during inference, identify suspicious visual tokens associated with the trigger, and suppress them before final generation. This makes PurMM especially suitable for deployed systems where retraining is expensive or infeasible.

## Why this paper matters

Fine-tuning-as-a-Service (FTaaS) makes MLLMs much easier to adapt to new tasks, but it also opens a realistic attack surface: poisoned downstream data can compromise a model while preserving strong performance on clean inputs.

This paper focuses on two core questions:

1. **Why are backdoor attacks in MLLMs both effective and hard to notice?**
2. **How can we defend against them at inference time without retraining?**

PurMM answers both questions by combining mechanistic analysis with a practical defense pipeline.

## Core insight

The paper identifies a phenomenon called **Attention Hijacking**.

When a backdoor trigger is present, the model places abnormally high attention on trigger-related visual regions and suppresses attention to the actual semantic content of the image. More importantly, the paper shows a **Hierarchical Mechanism**:

- In **shallow layers**, clean and backdoor samples behave similarly.
- In **middle and deep layers**, the trigger increasingly dominates attention.
- This explains why the model can still keep strong clean-task performance while being reliably hijacked when the trigger appears.

This observation is the foundation of PurMM.

## Method summary

PurMM is a **training-free, test-time backdoor purification framework**. Its pipeline contains three stages:

### 1. Attention-driven backdoor localization
PurMM extracts attention distributions from the model and focuses on image-token attention. Tokens with unusually high attention are treated as suspicious candidates for trigger regions.

### 2. Deep-Guided Filtering Mechanism (DGFM)
A direct attention-based selection can include many redundant tokens. To refine the localization, PurMM uses the observation that deeper layers are more informative for backdoor behavior. DGFM builds a reference set from deeper layers and keeps only nearby shallow-layer tokens, improving precision while preserving clean visual information as much as possible.

### 3. Backdoor token zeroing out
After suspicious tokens are refined, PurMM zeros out the corresponding visual embeddings and performs a second generation pass. This removes the trigger’s effect at inference time without changing model parameters.

## Experimental setting

The paper evaluates PurMM on two mainstream MLLMs:

- **LLaVA-v1.5-7B**
- **InternVL2.5-8B**

It reports results on three downstream tasks:

- **ScienceQA**
- **IconQA**
- **Flickr30k**

The backdoor setting follows a trigger-based poisoning protocol with LoRA fine-tuning. Evaluation includes:

- **CP (Clean Performance)**
- **ASR (Attack Success Rate)**
- **TP (Trade-off Performance)**
- **RP (Recovery Performance)**

## Main findings

PurMM consistently achieves the strongest balance between defending against backdoor behavior and preserving normal model utility.

### Representative results

On **LLaVA + ScienceQA**:

- **ASR drops from 99.55% to 0.84%**
- **TP increases from 43.86 to 91.90**

On **InternVL + ScienceQA**:

- **ASR drops from 97.47% to 8.53%**
- **TP reaches 90.90**

The paper also highlights **recovery capability**:

- On **ScienceQA**, **RP improves from 0.35% to 83.94%**
- On **IconQA**, **RP reaches 72.56%**
- On **Flickr30k**, **RP reaches 65.37%**

These results show that PurMM does more than block malicious outputs. It can often restore poisoned samples to normal, semantically correct responses.

## Robustness analysis

The paper further studies robustness under different settings.

### Different poisoning ratios
PurMM remains effective when the poisoning ratio changes from **5%** to **15%**, while keeping both RP and CP above 80% in the reported LLaVA-ScienceQA setting.

### Different trigger types
PurMM performs strongly on localized triggers such as:

- **Patch triggers**
- **Pixel triggers**
- **Logo triggers**

Among them, semantically integrated logo triggers are harder to purify, but PurMM still maintains strong overall robustness.

### Potential adaptive attacks
The paper also evaluates multi-trigger settings such as **Fixed Dual** and **Random Triple**, designed to disperse attention and evade attention-based defenses. PurMM remains effective under these settings, indicating strong resilience against plausible adaptive strategies.

## Key contributions

The paper makes three main contributions:

1. It reveals the mechanism of **Attention Hijacking** and the **hierarchical attention pattern** behind backdoor behavior in MLLMs.
2. It proposes **PurMM**, a practical **test-time** defense that requires **no retraining**.
3. It demonstrates strong effectiveness across different models, tasks, poisoning ratios, trigger types, and adaptive attack settings.

## Related Work

### 1. Backdoor Cleaning without External Guidance in MLLM Fine-tuning (NeurIPS 2025) [Paper Link](https://openreview.net/forum?id=os4QYDf3Ms)
### 2. Probing Semantic Insensitivity for Inference-Time Backdoor Defense in Multimodal Large Language Model (AAAI 2026) [Paper Link](https://ojs.aaai.org/index.php/AAAI/article/view/40891)

## Citation

### BibTeX

```bibtex
@inproceedings{jiang2026purmm,
  title     = {PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models},
  author    = {Jiang, Wenzheng and Liang, Ke and Rong, Xuankun and Zhou, Jingxuan and Zhong, Zhengyi and Wan, Guancheng and Wang, Ji},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {40},
  number    = {42},
  pages     = {35562--35570},
  year      = {2026},
  doi       = {10.1609/aaai.v40i42.40867}
}
```


