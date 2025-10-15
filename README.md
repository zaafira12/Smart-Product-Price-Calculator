 ## Smart Product Pricing Challenge — Amazon ML 2025
 # Overview

This project was developed for the Unstop Machine Learning Challenge 2025 – Smart Product Pricing.
The goal is to predict optimal product prices in an e-commerce environment using textual descriptions, product images, and quantitative details.

Products vary widely in brand, specifications, and quantity — making pricing prediction a multimodal regression task.
Evaluation metric: SMAPE (Symmetric Mean Absolute Percentage Error).

## Problem Statement

In e-commerce, determining an optimal selling price for a product is key for maximizing sales and profitability.
Given:

Product title, description, and quantity (text)

Product image URL (visual)

Price (target for training data)

The task is to build a machine learning model that can analyze both text and image modalities to predict the product’s market price.

## Dataset Description

The dataset was originally released by the Amazon ML Competition Group.
It contains:

Split	Samples	Description
train.csv	75,000	Training data with labeled prices
test.csv	75,000	Test data for evaluation
images/	~146,000	Product images (train + test combined)

Each record includes:

sample_id → Unique identifier

catalog_content → Product title + description + quantity text

image_link → URL to product image

price → Target variable (only in training data)

⚠️ The dataset is not uploaded due to size restrictions (146K images, >50GB).
Please refer to the Amazon ML Challenge dataset release for access.

## Model Architecture

This solution uses a multimodal ensemble that combines text embeddings, image embeddings, and manual numeric features.

🔹 1. Text Encoder (NLP)

Model: sentence-transformers/paraphrase-mpnet-base-v2

Extracts 768-dimensional embeddings from catalog_content

Token length capped at 128

🔹 2. Image Encoder (Vision)

Model: tf_efficientnet_b3_ns

Trained on ImageNet

Extracts 1536-dimensional embeddings from product images resized to 224×224

🔹 3. Manual Features

Derived from:

Word count / character length of description

Number of digits in text (quantity indicators)

Average brightness of the product image

Presence of brand names or specific keywords

🔹 4. Fusion + Ensemble

After extracting embeddings:

Text, image, and manual features are concatenated into a single feature vector.

Three ensemble regressors are trained:

LightGBM

XGBoost

CatBoost

Their predictions are averaged (weighted ensemble) to generate the final price.

## Training Details
Parameter	Value
Frameworks	PyTorch, Hugging Face Transformers, timm, LightGBM
Batch size	16
Image size	224×224
Token max length	128
Optimizer	AdamW
Learning rate	1e-5
Epochs	3 (feature extraction mode)
Cross-validation	5-fold
Metric	SMAPE (lower is better)

## Evaluation Metric

The Symmetric Mean Absolute Percentage Error (SMAPE) is used:

SMAPE=n1​∑(∣ytrue​∣+∣ypred​∣)/2∣ypred​−ytrue​∣​×100
Example:

Actual	Predicted	SMAPE
100	120	18.18%

## Pipeline Overview
┌─────────────────────┐
│ train.csv / images  │
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────┐
│ 1️⃣ Text Embeddings (SentenceTransformer) │
└──────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│ 2️⃣ Image Embeddings (EfficientNet-B3) │
└──────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 3️⃣ Manual Features │
└─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ 4️⃣ LightGBM / XGBoost / CatBoost │
└──────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Final Ensemble Output │
└─────────────────────┘

## How to Reproduce (Google Colab)

Mount Drive and install dependencies:

from google.colab import drive
drive.mount('/content/drive')

!pip install -q transformers timm lightgbm xgboost catboost scikit-learn pillow


Set dataset path:

DATA_PATH = "/content/drive/MyDrive/SmartPricingDataset"


Run the feature extraction and training notebooks (included in /notebooks folder).

## Results
Model	Validation SMAPE ↓
SentenceTransformer only	19.5
EfficientNet only	18.9
+ Manual Features	17.8
+ LightGBM Ensemble	16.2
+ Weighted Ensemble (XGB + LGBM + CatBoost)	15.7 (final)

The final ensemble achieved SMAPE = 15.7% on validation and ranked in the top 5% on the leaderboard.

## Future Improvements

Add CLIP or BLIP-2 multimodal transformers for joint text-image encoding

Fine-tune DeBERTa & EfficientNet end-to-end with mixed precision

Use quantile regression for price uncertainty estimation
