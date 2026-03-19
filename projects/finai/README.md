# FinAI — Real-Time Fraud Detection

Reproduction and experimentation based on:
> **"FinAI: Deep Learning for Real-Time Anomaly Detection in Financial Transactions"**
> Jaydeep Taralkar, 2025 — World Journal of Advanced Engineering Technology and Sciences, Vol. 15, Issue 02, pp. 454–461

## Research Question

Can a three-tiered deep learning architecture (stream processing + DL engine + self-adaptive learning) achieve real-time fraud detection at scale (>10K TPS, <50ms latency) while maintaining high precision/recall?

## Key Results from Paper

| Metric | Value |
|---|---|
| Throughput | 12,800 TPS |
| Avg Latency | 47 ms |
| Precision | 94.3% |
| Recall | 91.7% |
| Fraud Detected | $37.2M (missed by traditional systems) |
| Manual Review Reduction | 79% |

## Project Structure

```
finai/
├── research-state.yaml    # Hypotheses & status
├── research-log.md        # Decision timeline
├── findings.md            # Evolving narrative
├── study-guide.md         # Step-by-step reproduction guide
├── requirements.txt       # Python dependencies
├── literature/            # Papers & survey
│   ├── survey.md
│   └── WJAETS-2025-0354.pdf
├── src/                   # Reusable code
├── data/                  # Datasets & metrics
├── experiments/           # Per-hypothesis work
└── paper/                 # Final write-up
```
