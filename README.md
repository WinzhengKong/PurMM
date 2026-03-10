
# PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models

[![Venue](https://img.shields.io/badge/Venue-AAAI%202026-blue.svg)](https://aaai.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

[cite_start]This repository contains the official implementation of the AAAI 2026 accepted paper: **PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models**[cite: 1, 2, 22].

[cite_start]**Authors:** Wenzheng Jiang, Ke Liang, Xuankun Rong, Jingxuan Zhou, Zhengyi Zhong, Guancheng Wan, Ji Wang[cite: 3, 4].
[cite_start]**Affiliations:** National University of Defense Technology, Wuhan University[cite: 5, 6].

---

## 📌 Introduction

[cite_start]The downstream fine-tuning of Multimodal Large Language Models (MLLMs) is advancing rapidly, allowing general models to achieve superior performance on domain-specific tasks[cite: 9]. [cite_start]However, this open fine-tuning pipeline introduces critical vulnerabilities: attackers can easily poison datasets to implant backdoors into MLLMs[cite: 10, 43]. 

[cite_start]We conducted an in-depth investigation of backdoor attacks on MLLMs and revealed the core mechanism behind their effectiveness: **Attention Hijacking** and its **Hierarchical Mechanism**[cite: 11]. [cite_start]During an attack, models excessively focus on trigger regions[cite: 57]. [cite_start]Interestingly, attention for backdoor and clean samples is similar in shallow layers, but backdoor features elicit a significant attention shift in deeper layers, hijacking control to trigger targeted outputs[cite: 59].



[cite_start]Guided by this insight, we propose **PurMM**, an attention-guided test-time backdoor purification framework[cite: 12, 66].

## ✨ Key Features

[cite_start]Unlike existing defenses that require retraining or altering the model post-fine-tuning, PurMM operates purely at test-time[cite: 14, 52].

* [cite_start]**Training-Free Defense:** Eliminates backdoors without requiring retraining or modifications to the training process[cite: 14, 54].
* **Three-Stage Purification Pipeline:**
  1. [cite_start]**Attention-Driven Backdoor Localization:** Locates visual tokens exhibiting anomalous attention[cite: 12, 13, 61].
  2. [cite_start]**Deep-Guided Filtering Mechanism:** Filters localized tokens using deep-layer cues to preserve clean utility[cite: 13, 62, 210].
  3. [cite_start]**Zeroing Out:** Zeros out the corresponding components in the visual embeddings[cite: 13, 63].
* [cite_start]**High Recovery Performance:** Not only defends against targeted outputs but also successfully restores the correct answers for poisoned samples[cite: 12, 63, 67].

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and set up the environment:

```bash
git clone [https://github.com/yourusername/PurMM.git](https://github.com/yourusername/PurMM.git)
cd PurMM
conda create -n purmm python=3.9
conda activate purmm
pip install -r requirements.txt

```

### 2. Supported Models & Datasets

Our framework has been extensively evaluated on state-of-the-art MLLMs:

* 
**Models:** `LLaVA-v1.5-7B` , `InternVL2.5-8B` 


* 
**Datasets:** `ScienceQA` (VQA) , `IconQA` (VQA) , `Flickr30k` (Image Captioning) 



### 3. Usage

Run the test-time purification script on your fine-tuned model. *(Note: Replace placeholders with your local paths)*

```bash
python run_purmm.py \
    --model_path /path/to/fine-tuned-mllm \
    --dataset scienceqa \
    --apply_dgfm True

```

---

## 📊 Main Results

Extensive experiments show that PurMM sharply reduces attack success rates (ASR) while maintaining normal clean performance (CP), achieving an optimal trade-off performance (TP).

Comparison on Mainstream MLLMs 

| Model | Method | ScienceQA (CP↑ / ASR↓) | IconQA (CP↑ / ASR↓) | Flickr30k (CP↑ / ASR↓) |
| --- | --- | --- | --- | --- |
| **LLaVA** | Backdoor FT | 87.26 / 99.55 

 | 82.30 / 84.40 

 | 71.23 / 83.00 

 |
|  | **PurMM (Ours)** | <br>**84.63 / 0.84** 

 | <br>**80.07 / 4.04** 

 | <br>**66.22 / 5.40** 

 |
| **InternVL** | Backdoor FT | 97.92 / 97.47 

 | 97.21 / 93.07 

 | 47.84 / 31.17 

 |
|  | **PurMM (Ours)** | <br>**90.33 / 8.53** 

 | <br>**93.94 / 31.52** 

 | <br>**46.19 / 59.45** 

 |

(For comprehensive baselines including DiffPure, ZIP, SampDetox, and Sparse VLM, please refer to Table 1 in the paper.)

---

## 📑 Citation

If you find our work or this code useful in your research, please consider citing:

```bibtex
@inproceedings{jiang2026purmm,
  title={PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models},
  author={Jiang, Wenzheng and Liang, Ke and Rong, Xuankun and Zhou, Jingxuan and Zhong, Zhengyi and Wan, Guancheng and Wang, Ji},
  booktitle={Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2026}
}

```
