EMNIST Feature Extraction: Deep Belief Networks vs. Supervised Learning
Project Overview
This project explores the generative and discriminative power of Deep Belief Networks (DBN) using the EMNIST Letters dataset. The goal is to evaluate how unsupervised pre-training contributes to feature extraction, noise robustness, and hierarchical data representation.

Key Methodologies:
Unsupervised Feature Learning: We construct a DBN by stacking 3 Restricted Boltzmann Machines (RBMs). We visualize the Receptive Fields (weight matrices) to demonstrate how the model learns to detect edges, strokes, and letter components without labeled data.

Classification & Read-out: To perform class recognition, a linear read-out layer is applied to the final hidden layer. This setup tests the quality of the latent features extracted by the DBN.

Comparative Analysis: We benchmark the DBN's performance against a standard Supervised Neural Network to compare the efficiency of generative vs. discriminative training.

Noise Robustness: We evaluate the model's resilience by introducing varying levels of noise. We produce robustness curves to visualize the decay of accuracy as a function of data corruption.

Error Analysis: We generate a Confusion Matrix and apply Hierarchical Clustering to the model's errors. This allows us to visualize the "morphological similarity" the model perceives between different letter classes (e.g., how the model clusters 'i' and 'l' vs 'm' and 'n').

Technical Implementation
Environment: Developed on Kaggle using free GPU (Tesla P100/T4) for efficient RBM training.

Framework: Built using PyTorch.

Credits: Core DBN logic is adapted and modified from the flavio2018/Deep-Belief-Network-pytorch repository.