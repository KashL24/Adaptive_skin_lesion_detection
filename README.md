# Adaptive Query Allocation for Robust Skin Lesion Detection using RT-DETRv3

## Overview

This repository presents an enhanced version of **RT-DETRv3** for **skin lesion detection** by introducing **Adaptive Query Allocation** and **Adaptive Diversity Learning**. The proposed method dynamically allocates decoder queries based on lesion difficulty and encourages diverse query representations to improve localization of challenging skin lesions.

This work is developed using the **ISIC 2018 Skin Lesion Dataset** and is intended for research purposes.

---

## Motivation

RT-DETRv3 uses a fixed query allocation strategy for all objects. However, skin lesions exhibit significant variations in:

- Lesion size
- Border irregularity
- Boundary clarity
- Shape complexity

Using the same number of decoder queries for every lesion can result in inefficient supervision.

Our proposed method introduces **difficulty-aware adaptive query selection** to provide stronger supervision for challenging lesions while maintaining efficient training.

---

# Proposed Method

The proposed framework consists of three major components:

## 1. Lesion Difficulty Estimation

Each lesion is assigned a difficulty score based on three handcrafted features:

- Lesion Size
- Border Irregularity
- Boundary Clarity

The normalized features are combined to generate a **difficulty score** for every training image.

---

## 2. Adaptive Query Allocation

During training:

- Hungarian Matching assigns positive decoder queries.
- IoU is computed between matched predictions and ground truth.
- Positive queries are ranked according to IoU.
- The lesion difficulty determines the number of positive queries (**Adaptive Top-K**) used for supervision.

Hard lesions receive more decoder queries than easy lesions.

---

## 3. Adaptive Diversity Loss

To avoid redundant query representations:

- Decoder query embeddings are extracted.
- Selected positive embeddings are normalized.
- Pairwise cosine similarity is computed.
- Diversity loss encourages different positive queries to capture different lesion characteristics.

The diversity loss is weighted according to lesion difficulty.

---

# Training Pipeline

```
Input Image
      │
      ▼
Backbone (ResNet18)
      │
      ▼
Encoder
      │
      ▼
Transformer Decoder
      │
      ▼
950 Decoder Queries
      │
      ▼
Hungarian Matching
      │
      ▼
Matched Positive Queries
      │
      ▼
IoU Ranking
      │
      ▼
Difficulty Score
      │
      ▼
Adaptive Top-K Selection
      │
      ▼
Selected Query Embeddings
      │
      ▼
Cosine Similarity Diversity Loss
      │
      ▼
Difficulty Weighting
      │
      ▼
Final Adaptive Diversity Loss
      │
      ▼
Backpropagation
```

---

# Inference Pipeline

During inference, no difficulty score is required.

```
Input Image
      │
      ▼
Backbone
      │
      ▼
Encoder
      │
      ▼
Transformer Decoder
      │
      ▼
Bounding Box Predictions
      │
      ▼
Confidence Scores
      │
      ▼
Final Skin Lesion Detection
```

The adaptive strategy only influences training. The trained model performs standard inference without additional computational overhead.

---

# Features

- Difficulty-aware query allocation
- Adaptive Top-K supervision
- Cosine similarity diversity loss
- Difficulty-weighted diversity learning
- RT-DETRv3 based architecture
- End-to-end transformer detector
- Single-stage object detection

---

# Dataset

**ISIC 2018 Skin Lesion Dataset**

- Task: Lesion Detection
- Number of Classes: 1 (Lesion)
- Annotation Format: COCO

---

# Repository Structure

```
ppdet/
│
├── data/
│   └── source/
│       └── coco.py
│
├── modeling/
│   ├── heads/
│   │     └── detr_head.py
│   │
│   └── losses/
│         ├── detr_loss.py
│         └── diversity_loss.py
│
└── utils/
      └── difficulty_score.py
```

---

# Files Modified

- `ppdet/data/source/coco.py`
- `ppdet/utils/difficulty_score.py`
- `ppdet/modeling/losses/diversity_loss.py`
- `ppdet/modeling/losses/detr_loss.py`
- `ppdet/modeling/heads/detr_head.py`

---

# Methodology

1. Compute lesion difficulty score.
2. Load difficulty scores during training.
3. Pass difficulty scores into the loss function.
4. Obtain decoder query embeddings.
5. Perform Hungarian Matching.
6. Rank matched queries using IoU.
7. Select Adaptive Top-K positive queries.
8. Compute diversity loss using cosine similarity.
9. Weight diversity loss using lesion difficulty.
10. Optimize RT-DETRv3 using the proposed adaptive loss.

---

# Expected Advantages

- Better localization of difficult lesions
- Improved representation diversity
- Reduced redundant decoder queries
- Better supervision for challenging samples
- Potential improvement in AP and AP50

---

# Current Status

- [x] Difficulty score generation
- [x] Difficulty-aware dataset loader
- [x] Decoder embedding extraction
- [x] Adaptive Query Allocation
- [x] Adaptive Top-K selection
- [x] Adaptive Diversity Loss
- [x] Difficulty-weighted training
- [x] Integration into RT-DETRv3
- [x] Successful model training
- [ ] Extensive evaluation and benchmarking (ongoing)

---

# Future Work

- Evaluate on additional dermoscopy datasets.
- Investigate learnable difficulty estimation.
- Extend adaptive query allocation to multi-class medical detection.
- Explore adaptive query diversity for segmentation tasks.

---

# Citation

If you use this repository in your research, please cite the corresponding paper (to be added).

---

# Acknowledgements

This work is based on the RT-DETRv3 framework and extends it with adaptive query allocation and diversity learning for skin lesion detection.
