# Literature Survey

## Core Paper

### 1. FinAI: Deep Learning for Real-Time Anomaly Detection in Financial Transactions (Taralkar, 2025)
- **File**: [WJAETS-2025-0354.pdf](WJAETS-2025-0354.pdf)
- **Journal**: World Journal of Advanced Engineering Technology and Sciences (WJAETS), Vol. 15, Issue 02, pp. 454–461
- **Key results**: 12,800 TPS, 47ms latency, 94.3% precision, 91.7% recall
- **Architecture**: Three-tiered — Stream Processing (Kafka + Spark) → Deep Learning Engine → Self-Adaptive Learning Module
- **Deployment**: Cloudera distributed computing platform
- **Evaluation**: 6-month production evaluation; detected $37.2M in fraud missed by traditional systems
- **Relevance**: Primary study paper — focus on speed optimization and real-time inference

## Related Work (To Investigate)

- [ ] Real-time fraud detection with Apache Kafka and Spark Streaming — survey best practices
- [ ] Self-adaptive learning / online learning for concept drift in fraud detection
- [ ] Comparison with Transformer-based streaming fraud detection (D2 in paper list)
- [ ] NVIDIA TensorRT optimization for fraud detection inference
- [ ] Federated learning for privacy-preserving real-time fraud detection
