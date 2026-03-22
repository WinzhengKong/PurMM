<h1 align="center">PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models</h1>

## 📖 Abstract

Downstream fine-tuning of Multimodal Large Language Models (MLLMs) is advancing rapidly, allowing general models to achieve superior performance on domain-specific tasks. Yet most prior research focuses on performance gains and overlooks the vulnerability of the fine-tuning pipeline: attackers can easily poison the dataset to implant backdoors into MLLMs. We conduct an in-depth investigation of backdoor attacks on MLLMs and reveal the phenomenon of **Attention Hijacking** and its **Hierarchical Mechanism**. Guided by this insight, we propose PurMM, a **test-time backdoor purification framework** that removes visual tokens exhibiting anomalous attention, thereby avoiding targeted outputs while restoring correct answers. PurMM contains three stages: (1) locating tokens with abnormal attention, (2) filtering them using deep-layer cues, and (3) zeroing out their corresponding components in the visual embeddings. Unlike existing defences, PurMM dispenses with retraining and training-process modifications, operating at test-time to restore model performance while eliminating the backdoor. Extensive experiments across multiple MLLMs and datasets show that PurMM maintains normal performance, sharply reduces attack success rates, and consistently converts backdoor outputs to benign ones, offering a new perspective for safeguarding MLLMs.
## 🌟 Overview

This repository is based on the paper **PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models**, which investigates how backdoor attacks emerge in MLLMs and how they can be mitigated at inference time. The paper is motivated by the growing adoption of **Fine-tuning-as-a-Service (FTaaS)**, where users adapt general-purpose multimodal models to downstream tasks through lightweight fine-tuning. While this setting greatly improves usability, it also creates a realistic attack surface: poisoned downstream data can implant hidden malicious behaviors into the model while preserving strong performance on clean samples.

A central finding of the paper is that backdoor behavior in MLLMs is closely related to a phenomenon termed **Attention Hijacking**. When a trigger is present, the model allocates abnormally high attention to trigger-related visual regions and correspondingly suppresses attention to the true semantic content of the image. The paper further shows that this effect follows a **hierarchical mechanism** across layers: shallow layers behave similarly for both clean and poisoned samples, while middle and deep layers increasingly amplify the influence of the trigger. This explains why backdoored models can remain highly capable on standard inputs yet become reliably controlled when the trigger appears.

Building on this insight, PurMM introduces a practical purification framework for inference-time defense. Rather than retraining the model, cleaning data, or requiring external supervision, PurMM directly operates on attention patterns observed during generation. It first localizes suspicious image tokens through abnormal attention concentration, then refines these candidates with a **Deep-Guided Filtering Mechanism (DGFM)** that leverages the stronger backdoor signals found in deeper layers, and finally suppresses the suspicious visual embeddings through token zeroing before regenerating the output. In this way, PurMM removes the effect of the trigger while preserving as much clean semantic information as possible, making it particularly suitable for already deployed systems where retraining is expensive or infeasible.

## 🔬 Method

PurMM is designed as a **test-time backdoor purification framework** that does not alter model parameters and does not rely on additional clean annotations. Its overall design is grounded in the observation that the backdoor trigger gradually dominates attention as signals propagate through the model. Based on this, the method first inspects attention distributions over image tokens to locate regions that receive disproportionately high attention. Since a naive selection based only on attention magnitude can include many irrelevant tokens, PurMM further introduces **DGFM**, which uses deeper-layer attention as a more reliable reference and filters shallow-layer candidates according to their spatial proximity to these deeper suspicious regions. After obtaining a refined set of likely trigger-related tokens, PurMM zeros out the corresponding visual embeddings and performs a second generation pass. This process effectively weakens the trigger’s control over the model while preserving the model’s original ability to reason over the remaining visual content.

Compared with retraining-based defenses, PurMM is lightweight and deployment-friendly. Compared with data-cleaning approaches, it acts directly at inference time and can therefore be applied even when the original fine-tuning data or training process is unavailable. The method is especially compelling because it does not simply block malicious outputs; it aims to recover the model’s normal understanding of poisoned inputs and restore semantically correct responses.

## 🧪 Experiments

The paper evaluates PurMM on two representative MLLMs, **LLaVA-v1.5-7B** and **InternVL2.5-8B**, across three downstream tasks: **ScienceQA**, **IconQA**, and **Flickr30k**. The attack setting follows a trigger-based poisoning protocol under LoRA fine-tuning, and the evaluation reports **CP (Clean Performance)**, **ASR (Attack Success Rate)**, **TP (Trade-off Performance)**, and **RP (Recovery Performance)**. Across models and tasks, PurMM consistently delivers a strong balance between suppressing backdoor behavior and preserving normal utility.

On **LLaVA + ScienceQA**, PurMM reduces **ASR from 99.55% to 0.84%** while increasing **TP from 43.86 to 91.90**. On **InternVL + ScienceQA**, it lowers **ASR from 97.47% to 8.53%** and reaches a **TP of 90.90**. The paper also highlights the recovery capability of the method: on **ScienceQA**, **RP improves from 0.35% to 83.94%**; on **IconQA**, **RP reaches 72.56%**; and on **Flickr30k**, **RP reaches 65.37%**. These results suggest that PurMM not only prevents attacker-controlled outputs but can often restore poisoned samples to semantically correct predictions.

The robustness analysis further demonstrates that PurMM remains effective under different poisoning ratios, different localized trigger types such as **patch**, **pixel**, and **logo** triggers, and even under plausible adaptive attack settings including **Fixed Dual** and **Random Triple** triggers designed to disperse attention and evade attention-based defenses. Although semantically integrated triggers such as logos are more challenging, the method still maintains strong overall performance. Taken together, these experiments show that PurMM is a robust, general, and practical defense for backdoor threats in multimodal large language models.

## 📚 Related Work

### 1. Backdoor Cleaning without External Guidance in MLLM Fine-tuning (NeurIPS 2025)  
[Paper Link](https://openreview.net/forum?id=os4QYDf3Ms)

### 2. Probing Semantic Insensitivity for Inference-Time Backdoor Defense in Multimodal Large Language Model (AAAI 2026)  
[Paper Link](https://ojs.aaai.org/index.php/AAAI/article/view/40891)

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
