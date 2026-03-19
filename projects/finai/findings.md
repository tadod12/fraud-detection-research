# Findings — FinAI Real-Time Fraud Detection

## Current Understanding

FinAI (Taralkar, 2025) is a deep learning framework for real-time anomaly detection in financial transactions. It uses a three-tiered architecture deployed on Cloudera's distributed computing platform:

1. **Stream Processing Layer** — Apache Kafka for ingestion + Apache Spark for streaming analytics
2. **Deep Learning Engine** — Specialized neural network for fraud scoring
3. **Self-Adaptive Learning Module** — Continuous model updating to handle concept drift

Reported results over a 6-month evaluation: 12,800 TPS, 47ms average latency, 94.3% precision, 91.7% recall. Detected $37.2M in fraud that traditional systems missed and reduced manual review workloads by 79%.

## Patterns and Insights

- **Speed vs accuracy trade-off**: Real-time systems must process transactions in <100ms while maintaining high detection rates
- **Concept drift**: Fraud patterns evolve — static models degrade over time; self-adaptive learning addresses this
- **Stream processing**: Kafka + Spark is the dominant stack for high-throughput financial data pipelines

## Lessons and Constraints

*(To be updated as experiments proceed)*

## Open Questions

1. Can the architecture be reproduced with open-source tools and a synthetic dataset (e.g., PaySim)?
2. What is the minimum hardware requirement to achieve >1K TPS with acceptable latency?
3. How does the self-adaptive learning module compare to periodic full retraining?
4. What neural network architecture does the DL engine use — and can it be replaced with Transformer or GNN variants?
5. How does FinAI handle extreme class imbalance (fraud < 0.1% of transactions)?
