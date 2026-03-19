# FinAI Architecture — Study & Build Guide

> Steps to understand and reproduce the FinAI real-time fraud detection framework from:
> *"FinAI: Deep Learning for Real-Time Anomaly Detection in Financial Transactions"*
> (Taralkar, 2025, WJAETS Vol. 15)

---

## Phase 1: Literature & Understanding

> **Goal**: Understand the three-tiered architecture before building anything.

- **Step 1 — Read the paper thoroughly**
  - Focus on: architecture diagram, data flow, evaluation methodology
  - Map: problem statement → architecture → implementation → results

- **Step 2 — Study the three-tiered architecture**

  | Tier | Component | Technology | Purpose |
  |---|---|---|---|
  | **1. Stream Processing** | Data Ingestion | Apache Kafka | Real-time transaction ingestion at scale |
  | | Stream Analytics | Apache Spark Streaming | Feature extraction, windowed aggregation |
  | **2. DL Engine** | Neural Network | Custom DL model | Fraud scoring with attention mechanisms |
  | | Batch Normalization | Ghost BN | Generalization on mini-batches |
  | **3. Self-Adaptive Learning** | Online Learning | Incremental updates | Handle concept drift without retraining |

- **Step 3 — Search related literature**
  - Find 3–5 related real-time fraud detection papers
  - Compare: Kafka+Spark vs Kafka+Flink vs cloud-native solutions
  - Save summaries to `literature/`

- **Step 4 — Identify gaps & form hypotheses**
  - Can we reproduce the throughput claims with commodity hardware?
  - What DL architecture is optimal for the fraud scoring engine?

---

## Phase 2: Environment Setup

> **Goal**: Set up the streaming and ML infrastructure.

- **Step 5 — Install prerequisites**
  ```bash
  # Kafka + Spark (Docker or local)
  docker-compose up -d kafka spark

  # Python ML environment
  pip install -r requirements.txt
  ```

- **Step 6 — Set up Apache Kafka**
  - Create topics: `transactions`, `fraud-scores`, `alerts`
  - Configure producer for synthetic transaction generation
  - Configure consumer for fraud scoring pipeline

- **Step 7 — Set up Apache Spark Streaming**
  - Structured Streaming with Kafka source
  - Feature extraction in micro-batches
  - Windowed aggregation for behavioral features

---

## Phase 3: Data Preparation

> **Goal**: Create a realistic streaming fraud dataset.

- **Step 8 — Download or generate dataset**
  - Option A: PaySim synthetic dataset (Kaggle)
  - Option B: IEEE-CIS Fraud Detection dataset
  - Option C: Generate custom streaming data with Faker

- **Step 9 — Feature engineering for streaming**
  - Transaction amount, frequency, velocity features
  - Time-windowed aggregations (1min, 5min, 1hr)
  - Account-level behavioral profiles
  - Encoding categorical features for real-time processing

- **Step 10 — Build Kafka producer**
  - Simulate real-time transaction stream from dataset
  - Configurable TPS rate for benchmarking
  - Include fraud labels for evaluation

---

## Phase 4: Model Development

> **Goal**: Build and train the fraud detection DL engine.

- **Step 11 — Design the neural network**
  - Architecture options to test:
    - Feed-forward DNN with attention
    - 1D-CNN for feature patterns
    - Lightweight Transformer encoder
  - Key requirements: fast inference (<10ms per transaction), small model size

- **Step 12 — Train on batch data**
  - Initial training on historical labeled data
  - Handle class imbalance: SMOTE, focal loss, or class weights
  - Target: precision >94%, recall >91%

- **Step 13 — Optimize for inference speed**
  - ONNX export for fast inference
  - Model quantization (INT8)
  - Benchmark: latency per prediction, throughput on CPU vs GPU

---

## Phase 5: Streaming Pipeline Integration

> **Goal**: Connect DL model to the Kafka+Spark streaming pipeline.

- **Step 14 — Build the scoring pipeline**
  ```
  Kafka Producer → [transactions topic]
    → Spark Streaming (feature extraction)
    → DL Model (fraud scoring)
    → Kafka Consumer → [fraud-scores topic]
    → Alert System
  ```

- **Step 15 — Benchmark end-to-end performance**
  - Measure: TPS, p50/p95/p99 latency, precision, recall
  - Target: >1K TPS with <100ms p99 latency (scaled from paper's 12.8K TPS)
  - Test with increasing load to find breaking point

- **Step 16 — Implement self-adaptive learning (optional)**
  - Online learning: update model weights with new labeled data
  - Compare: full retrain vs incremental update vs replay buffer
  - Measure detection rate drift over simulated time periods

---

## Phase 6: Analysis & Documentation

> **Goal**: Evaluate and document the system.

- **Step 17 — Compare with baselines**

  | Baseline | Type |
  |---|---|
  | Rule-based system | Simple thresholds |
  | Batch XGBoost | Traditional ML, no streaming |
  | Batch TabNet | DL, no streaming (from existing project) |
  | FinAI pipeline | Streaming DL (this project) |

- **Step 18 — Document findings**
  - Performance comparison: speed vs accuracy trade-offs
  - Architecture decisions and lessons learned
  - Visualizations: latency distributions, ROC curves, throughput graphs

- **Step 19 — Generate final report**
  - Can the paper's results be reproduced?
  - What modifications improve performance?
  - Is real-time DL fraud detection practical on commodity hardware?

---

## Quick Start Code Snippet

```python
# Install
# pip install kafka-python pyspark torch scikit-learn imbalanced-learn pandas

from kafka import KafkaProducer, KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window
import torch
import json

# 1. Kafka Producer — Simulate transaction stream
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_transaction(tx):
    producer.send('transactions', value=tx)

# 2. Spark Streaming — Feature extraction
spark = SparkSession.builder \
    .appName("FinAI-FraudDetection") \
    .getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("subscribe", "transactions") \
    .load()

# 3. DL Model — Fraud scoring
class FraudScorer(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 4. Scoring pipeline
model = FraudScorer(input_dim=30)
model.eval()

# Process each micro-batch
# (integrate with Spark foreachBatch or use Kafka consumer directly)
```

---

## References

| Resource | Link |
|---|---|
| FinAI Paper | WJAETS Vol. 15, Issue 02, 2025 |
| Apache Kafka | [kafka.apache.org](https://kafka.apache.org/) |
| Apache Spark Streaming | [spark.apache.org](https://spark.apache.org/streaming/) |
| PaySim Dataset | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| ONNX Runtime | [onnxruntime.ai](https://onnxruntime.ai/) |

---

*Guide derived from skills: `0-autoresearch-skill`, `01-model-architecture`, `12-inference-serving`, `20-ml-paper-writing`*
