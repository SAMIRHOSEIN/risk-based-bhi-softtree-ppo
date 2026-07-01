# Risk-Based BHI Soft-Tree PPO

Interpretable reinforcement learning for bridge infrastructure management using Bridge Health Index (BHI), Proximal Policy Optimization (PPO), and soft decision tree policies.

## Overview

This repository provides an actively developed research framework for studying long-term bridge maintenance and replacement decisions under uncertainty.

The project combines civil infrastructure modeling, reinforcement learning, and interpretable machine learning. It models bridge condition states, deterioration, maintenance actions, replacement actions, intervention costs, and Bridge Health Index (BHI). It uses PPO to learn sequential decision-making policies and investigates soft decision tree actors to improve interpretability compared with black-box neural network policies.

This repository is part of ongoing PhD research in civil infrastructure systems, reinforcement learning, and interpretable machine learning.

## Why This Matters

Civil infrastructure agencies must make long-term maintenance and replacement decisions under uncertainty, limited budgets, and changing bridge conditions. Traditional optimization methods can be difficult to adapt, while black-box machine learning models can be difficult to explain to engineers and public decision-makers.

This project aims to provide reusable open-source research software for:

* bridge asset management,
* infrastructure resilience,
* life-cycle decision-making,
* interpretable reinforcement learning,
* risk-based maintenance planning,
* reproducible civil engineering AI research.

The goal is to make AI-based infrastructure decision-support methods more transparent, reproducible, and accessible to researchers and practitioners.

## Main Features

* Bridge Health Index-based environment for infrastructure decision-making
* Condition-state modeling for multiple bridge elements
* Deterioration transition modeling
* Maintenance and replacement action space
* Cost-aware reward structure
* PPO-based reinforcement learning workflow
* Soft decision tree actor architecture
* Policy validation and interpretation utilities
* Research-oriented framework for experimentation and extension

## Repository Status

This repository is under active development.

The current public version focuses on the reusable environment, PPO training pipeline, interpretable soft-tree actor architecture, bridge health modeling, action modeling, and validation utilities.

Full experimental results, publication-quality figures, and dissertation-scale case studies will be added as the research matures.

## Research Context

The project is motivated by the following question:

How can uncertain, heterogeneous, and evolving bridge condition information be converted into transparent and reliable maintenance decisions?

This repository explores that question using reinforcement learning and interpretable machine learning. Instead of learning only a black-box policy, the framework investigates soft decision tree policies that can provide more understandable decision structures for infrastructure management.

## Method Summary

At a high level, the framework includes:

1. A bridge environment representing bridge elements, condition states, deterioration, and actions.
2. A Bridge Health Index calculation used to summarize bridge-level condition.
3. A reward structure that balances bridge condition improvement and maintenance/replacement cost.
4. PPO training for sequential policy learning.
5. Soft decision tree actors for interpretable policy representation.
6. Validation tools for analyzing learned policies and decision paths.

## Installation

Clone the repository:

```bash
git clone https://github.com/SAMIRHOSEIN/risk-based-bhi-softtree-ppo.git
cd risk-based-bhi-softtree-ppo
```

Install dependencies:

```bash
pip install -r requirements.txt
```


## Applications

This framework can support research in:

* bridge maintenance planning,
* infrastructure asset management,
* life-cycle infrastructure optimization,
* reinforcement learning for civil engineering,
* interpretable AI for engineering decision-making,
* risk-based infrastructure management,
* transportation infrastructure resilience.

## Related Open-Source Work

I also maintain related scientific machine learning code, including Physics-Informed Neural Network workflows for cylindrical transport modeling. Together, these projects support reproducible AI research for civil and environmental engineering applications.

## Citation

If you use this repository in research, please cite the repository and related publications when available.

```bibtex
@software{moayedi_risk_based_bhi_softtree_ppo,
  author = {Moayedi, Amir},
  title = {Risk-Based BHI Soft-Tree PPO},
  year = {2026},
  url = {https://github.com/SAMIRHOSEIN/risk-based-bhi-softtree-ppo}
}
```

## License

This project is released under the MIT License.
