# ML4Physics – Homework 2: Regression of Anomalous Exponent α

This repository contains my solution to **Task 2** of the ML4Physics homework:
regression of the anomalous diffusion exponent **α** from stochastic trajectories
generated via Fractional Brownian motion.

The goal is to train a neural network that, given a 1D trajectory of length `T`,
predicts a continuous anomalous exponent `α ∈ [0, 2]`.  
The model is evaluated using **Mean Squared Error (MSE)**.

---

## Project structure

```text
ML4Physics_homework2/
│
├─ data/
│   ├─ regression_input.pt    # Tensor of shape (N, T): trajectories
│   └─ regression_true.pt     # Tensor of shape (N,): true α for each trajectory
│
├─ src/  (names may vary, adjust to your layout)
│   ├─ model.py               # Definition of the `model` class used for submission
│   ├─ train.py               # TrainingAlgorithm and training script
│   └─ utils.py               # Data loading utilities (train/val split, dataloaders)
│
└─ notebooks/
    └─ exploration.ipynb      # (optional) experiments, plots, evaluation