# Multimodal Architectures

A deep learning project exploring multimodal models that combine text and images for classification, fusion, and representation learning.

## Overview

This project implements various multimodal learning techniques using the **MM-IMDB dataset**, which combines movie posters (images) with plot summaries (text) for movie genre classification.

## Dataset

- **MM-IMDB**: Multimodal IMDB dataset combining movie posters and plot descriptions
- Dataset: https://www.innovatiana.com/en/datasets/mm-imdb-multimodal-imdb-dataset

## Key Components

### 1. Multimodal Embeddings
- Image embeddings using pretrained vision models (e.g., ResNet, ViT)
- Text embeddings using language models (e.g., BERT, RoBERTa)
- Joint representation learning

### 2. Fusion Strategies
- **Early Fusion**: Concatenating raw features before classification
- **Late Fusion**: Combining predictions from separate modality-specific models
- Comparison of fusion effectiveness for genre classification

### 3. Semi-Supervised Learning
- Pseudo-labeling for unlabeled data
- Leveraging large amounts of unannotated movie data

### 4. Self-Supervised Learning
- Consistency training across modalities
- Contrastive learning objectives
- Modality dropout experiments

## Techniques

- Multimodal embeddings
- Early and late fusion architectures
- Pseudo-labeling
- Consistency training
- Cross-modal representation learning

## Research

Project documentation and report: [Typst](https://typst.app/project/ra2kU2K5bMHrj9o9C3bu8W)

## Course

This project is part of **IKT469 - Deep Neural Networks** at the University of Agder Norway (UIA).
