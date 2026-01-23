# GRL-DEACO: Graph-Aware RL for Adaptive ACO Hyperparameter Tuning in 3D Petrochemical Pipe Routing

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green) <!-- change if needed -->
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

GRL-DEACO is a graph-aware reinforcement learning framework for **adaptive ACO hyperparameter tuning** in **3D petrochemical pipe routing** under strict engineering constraints. It integrates:
- a **3D grid routing environment** with collision/clearance/bend-radius constraints,
- an **equipment topology graph** encoded by a multi-head **Graph Attention Network (GAT)**,
- a **cross-attention fusion** between graph embedding and compact connection-state features,
- a **PPO Actor–Critic** agent that outputs **continuous ACO hyperparameters**,
- a **green DEACO objective** that fuses geometric quality with surrogate **energy / CO₂** models,
- a web-based 3D digital factory platform for end-to-end validation.

> **Paper abstract** is included in the manuscript directory (see below).
