# 📝 Automated Essay Scoring System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-yellow.svg)](https://huggingface.co/docs/transformers)

A deep learning-based automated essay scoring system that uses fine-tuned BERT to evaluate essays and assign scores on a 1-6 scale. Achieves **89.5% accuracy** with threshold-based rounding optimization.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Optimization](#model-optimization)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🎯 Overview

This project implements an automated essay scoring system using BERT-base-uncased fine-tuned for regression tasks. The model reads student essays and predicts holistic scores (1-6) based on writing quality, coherence, and content relevance.

### How It Works
1. **Text Preprocessing**: Cleans and normalizes essay text
2. **Tokenization**: Handles long documents using sliding windows with stride
3. **Prediction**: BERT model outputs raw scores
4. **Rounding**: Configurable thresholds (best at 0.6) convert to final scores

## ✨ Key Features

- ✅ **High Accuracy**: 89.5% weighted F1 score with optimal rounding
- ✅ **Long Document Support**: Handles essays exceeding 512 tokens via stride-based chunking
- ✅ **Model Optimization**: 50% size reduction using FP16 (418MB → 209MB)
- ✅ **Flexible Rounding**: Multiple strategies with adjustable thresholds
- ✅ **Comprehensive Testing**: 10% holdout validation set
- ✅ **Production Ready**: Simple inference API with confidence scoring

## 📊 Dataset

**Source**: AES2 Essay Scoring dataset from Hugging Face Datasets

| Split | Size |
|-------|------|
| Training | 15,576 essays |
| Testing | 1,731 essays |

**Score Distribution**: Holistic scores ranging from 1 (lowest) to 6 (highest)

## 🏗️ Model Architecture
Input Essay
↓
[Text Cleaning] → Lowercase + remove punctuation
↓
[Tokenization] → BERT tokenizer (max_length=512, stride=256)
↓
[BERT-base-uncased] → 110M parameters
↓
[Regression Head] → Single neuron output
↓
Raw Score (1-6 range)
↓
[Rounding] → Threshold-based (optimal: 0.6)
↓
Final Score

## 📈 Results

### Performance Metrics (Best Model)

| Metric | Value |
|--------|-------|
| **Accuracy** | **89.5%** |
| **Weighted F1** | **0.895** |
| **MSE** | **0.110** |

### Classification Report (Threshold 0.6)

| Score | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 1 | 0.95 | 0.89 | 0.92 | 125 |
| 2 | 0.88 | 0.93 | 0.90 | 447 |
| 3 | 0.91 | 0.89 | 0.90 | 657 |
| 4 | 0.90 | 0.88 | 0.89 | 377 |
| 5 | 0.83 | 0.81 | 0.82 | 110 |
| 6 | 0.56 | 1.00 | 0.71 | 15 |

### Rounding Strategy Comparison

| Strategy | Accuracy | MSE | F1 Weighted |
|----------|----------|-----|--------------|
| Threshold 0.6 | **0.895** | **0.110** | **0.894** |
| Standard Round | 0.879 | 0.127 | 0.881 |
| Floor | 0.672 | 0.336 | 0.674 |
| Ceil | 0.341 | 0.683 | 0.352 |

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended, but CPU works)
