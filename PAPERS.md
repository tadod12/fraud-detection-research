# Advanced Fraud Detection Papers — Curated List

> Pick the papers you'd like me to set up as projects in the `projects/` folder using the repository templates.

Your existing project covers **TabNet** (Arık & Pfister, 2019) applied to bank fraud detection. The papers below go deeper into **complex models**, **fine-tuning**, **accuracy optimization**, and **speed/inference optimization**.

---

## Category A: Transformer & Graph Neural Network Models

These use the most complex architectures — graph-based reasoning, self-attention, and hybrid GNN+Transformer designs.

| # | Rank | Paper | Year | Key Idea |
|---|---|---|---|---|
| **A1** | Preprint | **Credit Card Fraud Detection Using Advanced Transformer Model** | 2024 | Adapts Transformer self-attention for tabular fraud data; F1 0.998, outperforms XGBoost, TabNet, NNs |
| **A2** | Q1 | **FraudGT: Graph Transformer for Financial Fraud Detection** | 2024 | GNN edge-based message passing + Transformer attention biases; 8–17% higher F1, 2.4x throughput (ICAIF 2024) |
| **A3** | N/R | **Graph-Enhanced Transformer Network (GETNet)** | 2025 | Hybrid GNN + Transformer on PaySim; 99.5% accuracy (IJRCMS) |
| **A4** | Preprint | **Financial Fraud Detection using Jump-Attentive GNN (JA-GNN)** | 2024 | Attention + efficient neighborhood sampling for anti-camouflage fraud detection |
| **A5** | Preprint | **FraudCoT: Chain-of-Thought Distillation for Graph-Based Fraud Detection** | 2026 | LLM-enhanced GNN with CoT distillation — 8.8% AUPRC improvement, 1,066x speedup |

---

## Category B: Hybrid Deep Learning Ensemble Models

These combine multiple model architectures (CNN, LSTM, Transformer, XGBoost) in ensemble/stacking for maximum accuracy.

| # | Rank | Paper | Year | Key Idea |
|---|---|---|---|---|
| **B1** | Q1 | **Hybrid DL Ensemble: CNN + LSTM + Transformer + XGBoost Meta-Learner** | 2024 | Stacking ensemble; sensitivity 0.961, specificity 0.999, AUC-ROC 0.972 (IEEE Access) |
| **B2** | Preprint | **Financial Fraud Detection Using Explainable AI and Stacking Ensemble** | 2025 | XGBoost + LightGBM + CatBoost + SHAP/LIME; 99% accuracy, AUC-ROC 0.99 on IEEE-CIS |
| **B3** | Preprint | **Ensemble KAN-XGBoost Model for Fraud Detection** | 2025 | Kolmogorov-Arnold Networks + XGBoost; 99% metrics on PaySim with SMOTE |
| **B4** | Q1 | **Credit Payment Fraud Detection: TabNet + XGBoost** | 2022 | Hybrid TabNet + XGBoost — extends your current TabNet project (IEEE ICCECE) |

---

## Category C: Generative Models for Imbalanced Data

These address the critical class imbalance problem using VAE, GAN, and hybrid generative approaches.

| # | Rank | Paper | Year | Key Idea |
|---|---|---|---|---|
| **C1** | Q2 | **Hybrid VAE + Graph Attention Network + XGBoost Stacking** | 2025 | VAE anomaly detection + GAT relational learning + stacking; tested on European CC and IEEE-CIS (MDPI AppliedMath) |
| **C2** | Q1 | **GA-GAN: Gated Attention GAN for Credit Card Fraud** | 2025 | GAN + gated attention generates high-quality synthetic fraud samples (PeerJ CS) |
| **C3** | N/R | **CGAN for Tabular Fraud Data Augmentation** | 2025 | CGANs generate synthetic transactions for extreme imbalance; LightGBM classifier (Tulipa) |
| **C4** | Q4 | **VAE for Unsupervised Fraud Detection in Financial Transactions** | 2025 | VAE-only anomaly detection via reconstruction error; compared with hybrid VAE+classifier (CEUR DECaT) |

---

## Category D: Speed Optimization & Real-Time Inference

These focus on making fraud detection models fast enough for production — low latency, edge deployment, model compression.

| # | Rank | Paper | Year | Key Idea |
|---|---|---|---|---|
| **D1** | N/R | **FinAI: Deep Learning for Real-Time Anomaly Detection in Financial Transactions** | 2025 | 12,800 TPS, 47ms latency; Kafka + Spark + self-adaptive learning (WJAETS) |
| **D2** | N/R | **Transformer-Based Financial Fraud Detection with Real-Time Cloud Streaming** | 2025 | Transformer in Kafka + Cloud Dataflow pipeline; captures temporal dependencies |
| **D3** | Preprint | **Generative Pretraining at Scale: Transformer-Based Encoding for Fraud Detection** | 2023 | GPT architecture for unsupervised feature learning + differential convolution for payments |
| **D4** | Q1 | **Efficient Financial Fraud Detection on Mobile Using Lightweight LLMs** | 2025 | Fine-tuned Llama-160M + Qwen-0.5B with LoRA/QLoRA; 99.47% accuracy, 168MB footprint (ACL) |

---

## How to Choose

| Priority | Recommended Papers |
|---|---|
| 🔗 Extend your **TabNet** project | **B4** — TabNet + XGBoost hybrid |
| 🧠 Most **complex / cutting-edge** model | **A2** (FraudGT) · **A3** (GETNet) · **A5** (FraudCoT) |
| 🎯 Highest **accuracy** claims | **B2** (99% acc) · **B1** (AUC 0.972) · **A1** (F1 0.998) |
| ⚖️ Solve **class imbalance** | **C1** (VAE+GAT+XGBoost) · **C2** (GA-GAN) |
| ⚡ **Production speed** / real-time | **D1** (FinAI 12.8K TPS) · **D4** (on-device LLM) |
| 📖 Best **venue quality** (Q1) | **A2** · **B1** · **B4** · **C2** · **D4** |
