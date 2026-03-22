<h1 align="center">PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models</h1>

[![Conference](https://img.shields.io/badge/AAAI-2026-brightgreen)](https://aaai.org/conference/aaai/aaai-26/)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

## 📖 Abstract

Downstream fine-tuning of Multimodal Large Language Models (MLLMs) is advancing rapidly, allowing general models to achieve superior performance on domain-specific tasks. Yet most prior research focuses on performance gains and overlooks the vulnerability of the fine-tuning pipeline: attackers can easily poison the dataset to implant backdoors into MLLMs. We conduct an in-depth investigation of backdoor attacks on MLLMs and reveal the phenomenon of **Attention Hijacking** and its **Hierarchical Mechanism**. Guided by this insight, we propose PurMM, a **test-time backdoor purification framework** that removes visual tokens exhibiting anomalous attention, thereby avoiding targeted outputs while restoring correct answers. PurMM contains three stages: (1) locating tokens with abnormal attention, (2) filtering them using deep-layer cues, and (3) zeroing out their corresponding components in the visual embeddings. Unlike existing defences, PurMM dispenses with retraining and training-process modifications, operating at test-time to restore model performance while eliminating the backdoor. Extensive experiments across multiple MLLMs and datasets show that PurMM maintains normal performance, sharply reduces attack success rates, and consistently converts backdoor outputs to benign ones, offering a new perspective for safeguarding MLLMs.

## 🌟 Overview

The paper is motivated by the growing adoption of **Fine-tuning-as-a-Service (FTaaS)**, where users adapt general-purpose multimodal models to downstream tasks through lightweight fine-tuning. While this setting greatly improves usability, it also creates a realistic attack surface: poisoned downstream data can implant hidden malicious behaviors into the model while preserving strong performance on clean samples. A central finding of the paper is that backdoor behavior in MLLMs is closely related to a phenomenon termed **Attention Hijacking**. When a trigger is present, the model allocates abnormally high attention to trigger-related visual regions and correspondingly suppresses attention to the true semantic content of the image. The paper further shows that this effect follows a **hierarchical mechanism** across layers: shallow layers behave similarly for both clean and poisoned samples, while middle and deep layers increasingly amplify the influence of the trigger. This explains why backdoored models can remain highly capable on standard inputs yet become reliably controlled when the trigger appears.

Building on this insight, **PurMM introduces a practical purification framework for inference-time defense**. Rather than retraining the model, cleaning data, or requiring external supervision, PurMM directly operates on attention patterns observed during generation. It first localizes suspicious image tokens through abnormal attention concentration, then refines these candidates with a **Deep-Guided Filtering Mechanism (DGFM)** that leverages the stronger backdoor signals found in deeper layers, and finally suppresses the suspicious visual embeddings through token zeroing before regenerating the output. In this way, PurMM removes the effect of the trigger while preserving as much clean semantic information as possible, making it particularly suitable for already deployed systems where retraining is expensive or infeasible.


## 📚 Relevant Projects

### 1. Backdoor Cleaning without External Guidance in MLLM Fine-tuning (NeurIPS 2025)  
[[Paper Link](https://openreview.net/forum?id=os4QYDf3Ms)]  [[Code Link](https://github.com/XuankunRong/BYE)]

### 2. Probing Semantic Insensitivity for Inference-Time Backdoor Defense in Multimodal Large Language Model (AAAI 2026)  
[[Paper Link](https://ojs.aaai.org/index.php/AAAI/article/view/40891)]

## 📝 Citation

Please kindly cite this paper in your publications if it helps your research:

```bibtex
@inproceedings{jiang2026purmm,
  title={PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models},
  author={Jiang, Wenzheng and Liang, Ke and Rong, Xuankun and Zhou, Jingxuan and Zhong, Zhengyi and Wan, Guancheng and Wang, Ji},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={42},
  pages={35562--35570},
  year={2026}
}
