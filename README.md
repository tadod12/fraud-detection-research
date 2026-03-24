# Research Repository

A multi-project research workspace for AI/ML paper study, experimentation, and writing.

## Structure

```
├── projects/                   # Individual research projects
│   └── {project-name}/
│       ├── research-state.yaml # Hypotheses, status, direction
│       ├── research-log.md     # Decision timeline
│       ├── findings.md         # Evolving narrative synthesis
│       ├── literature/         # Papers, summaries, survey
│       ├── src/                # Reusable project code
│       ├── data/               # Datasets, metrics
│       ├── experiments/        # Per-hypothesis work
│       │   └── {experiment}/
│       │       ├── protocol.md # What, why, prediction
│       │       ├── code/
│       │       └── results/
│       └── paper/              # Final write-up (LaTeX)
│
├── templates/                  # Project & experiment templates
│   ├── new-project/            # Scaffold for a new project
│   └── new-experiment/         # Scaffold for a new experiment
│
├── shared/                     # Cross-project utilities
│   └── utils/                  # Shared plotting, data loading, etc.
│
└── .agent/skills/              # 22 AI research skills
```

## Active Projects

| Project | Status | Focus |
|---|---|---|
| [tabnet](projects/tabnet/) | Bootstrap | TabNet for bank fraud detection |

## Quick Start — New Project

1. Copy `templates/new-project/` → `projects/{your-project-name}/`
2. Edit `research-state.yaml` with your research question and hypotheses
3. Add papers to `literature/` and start `literature/survey.md`
4. For each experiment: copy `templates/new-experiment/` → `experiments/{name}/`
5. Follow the two-loop architecture: **Inner Loop** (experiment) ↔ **Outer Loop** (synthesize)

## Skills Available

22 AI research skills in `.agent/skills/` — covering model architecture, fine-tuning, data processing, evaluation, paper writing, research ideation, and more.
