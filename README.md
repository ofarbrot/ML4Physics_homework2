# ML4Physics – Homework 2: Regression of Anomalous Exponent α

This repository contains our solution to **Task 2** of the ML4Physics homework:
regression of the anomalous diffusion exponent **α** from stochastic trajectories
generated via Fractional Brownian motion.

The goal is to train a neural network that, given a 1D trajectory of length `T`,
predicts a continuous anomalous exponent `α ∈ [0, 2]`.  
The model is evaluated using **Mean Squared Error (MSE)**.