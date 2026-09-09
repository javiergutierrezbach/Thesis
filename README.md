
# Formal Verification of Reinforcement Learning Controllers

This repository contains the code developed for my undergraduate honors thesis in Computer Science at Amherst College.

The project investigates how to formally verify reinforcement learning controllers operating in environments with unknown or black-box dynamics. The approach learns surrogate neural network models from sampled interactions with the underlying system and uses Lyapunov Barrier certificates to reason about controller safety and stability.

A key part of the work explores Lipschitz-continuity-based bounds for extending guarantees established on the learned surrogate model to the true underlying dynamics.

## Experiments

The framework was evaluated on several control problems, including:

- 2D docking
- Inverted-pendulum stabilization
- Quadrotor hovering

These experiments were used to study the effectiveness and robustness of the verification approach across different dynamical systems.

## Methods

The project combines techniques from:

- Reinforcement learning
- Neural network modeling
- Formal verification
- Lyapunov-based analysis
- Lipschitz continuity and robustness analysis

## About

This work was completed as my honors thesis in Computer Science at Amherst College under the supervision of Professor Haoze Wu.
