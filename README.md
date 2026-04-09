# Mechanistic Interpretability of LLM Behaviours via Transcoders

**Author:** Iuliia Vitiugova
**Affiliation:** DAMTP, University of Cambridge — MPhil Dissertation
**Model:** Qwen3-4B (base for transcoder work; instruct for baseline evaluation only)
**Inspired by:** [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Anthropic, 2025)

---

## Overview

This project investigates whether the circuit-level interpretability methods pioneered by Anthropic on Claude 3.5 Haiku can be reproduced on an **open-source model** (Qwen3-4B). We use **transcoders** (sparse autoencoders trained on MLP activations) to extract interpretable features, build **attribution graphs** tracing causal interactions between features across layers, and validate circuits through three types of **causal interventions**: ablation, activation patching, and feature steering.

The project introduces a formal definition of **behaviour** as a unit of analysis, a **four-type typology** of LLM computations, and an intervention-based **causal edge pipeline** that identifies which feature-to-feature connections are functionally causal (not just correlated).

---

---

## References

- Lindsey, J., Gurnee, W., et al. (2025). *On the Biology of a Large Language Model*. Anthropic Transformer Circuits Thread. [Link](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- Qwen Team (2025). *Qwen3 Technical Report*. [Link](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread.
- Cunningham, H., et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*.
