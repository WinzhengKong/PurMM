<h1 align="center">PurMM: Attention-Guided Test-Time Backdoor Purification in Multimodal Large Language Models</h1>

[AAAI'26 Oral]

<div align="center">
<img alt="method" src="PurMM_Framework.png">
</div>

<h2> 🙌 Abstract </h2>

Downstream fine-tuning of Multimodal Large Language Models (MLLMs) is advancing rapidly, allowing general models to achieve superior performance on domain-specific tasks. Yet most prior research focuses on performance gains and overlooks the vulnerability of the fine-tuning pipeline: attackers can easily poison the dataset to implant backdoors into MLLMs. We conduct an in-depth investigation of backdoor attacks on MLLMs and reveal the phenomenon of \textbf{Attention Hijacking} and its \textbf{Hierarchical Mechanism}. Guided by this insight, we propose \textbf{PurMM}, a \textbf{test-time backdoor purification} framework that removes visual tokens exhibiting anomalous attention, thereby avoiding targeted outputs while restoring correct answers. PurMM contains three stages: (1) locating tokens with abnormal attention, (2) filtering them using deep-layer cues, and (3) zeroing out their corresponding components in the visual embeddings. Unlike existing defences, PurMM dispenses with retraining and training-process modifications, operating at test-time to restore model performance while eliminating the backdoor. Extensive experiments across multiple MLLMs and datasets show that PurMM maintains normal performance, sharply reduces attack success rates, and consistently converts backdoor outputs to benign ones, offering a new perspective for safeguarding MLLMs.

