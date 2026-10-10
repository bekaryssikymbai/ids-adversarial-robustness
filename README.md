# Robustness of Tree-Based ML Models for Network Intrusion Detection under Gradient-Based Adversarial Attacks

## Authors
- Bekarys Sikymbay

## Research Question
How do gradient-based adversarial attacks (FGSM, PGD) affect the robustness of tree-based ensemble models (XGBoost, Random Forest) in network intrusion detection, and which model is more resilient?

## Overview
This repository contains the experimental pipeline for evaluating the adversarial robustness of XGBoost and Random Forest classifiers on the CIC-IDS2017 dataset. The pipeline includes:
- Data preprocessing
- Baseline model training
- Surrogate model for gradient generation
- FGSM and PGD adversarial attacks
- Statistical analysis and visualization

## System Requirements
- Python 3.10+
- 16 GB RAM
- CPU-only (no GPU required)
- OS: Linux / macOS / Windows

## Repository Structure
ids-adversarial-robustness/
├── data/sample/ # Micro-sample for pipeline testing
├── src/ # Source code
├── configs/ # Configuration files
├── notebooks/ # Jupyter notebooks for analysis
├── docs/ # Documentation
├── environment.yml # Conda environment with pinned versions
├── config.yaml # Experiment configuration
└── LICENSE



