# Transformer Playground

A collection of transformer-related implementations, notes, and experiments—from the original sequence-to-sequence architecture to vision transformers and multimodal models.

---

## Subdirectories

### [BERT](BERT/)

**Bidirectional Encoder Representations from Transformers** — Pre-training the Transformer encoder with two objectives: masked language modeling and next-sentence prediction. Notes cover the 2018 paper, parameter counts (Base ~110M, Large ~340M), pretraining vs fine-tuning, and data/training details (e.g., 15% random masking, Wikipedia-scale data). Includes lecture and walkthrough references.

### [transformer](transformer/)

**Original Transformer** — Implementation notes for the “Attention is All You Need” sequence-to-sequence model. Covers input embeddings (word indexing, learned embeddings), sinusoidal positional encoding, layer normalization, and the feed-forward block (position-wise MLP, hidden dimension, code snippets). Focus is on the encoder/decoder building blocks rather than full training code.

### [vision_transformer](vision_transformer/)

**Vision Transformer (ViT)** — Pure transformer for image classification: treat an image as a sequence of 16×16 patches (e.g., 196 patches + CLS for 224×224). Notes on the original ViT paper, inductive bias vs CNNs, large-scale pretraining (ImageNet-21K, JFT-300M), and implementation details (patch embedding, position embeddings, multi-head attention dimensions). Also summarizes **MAE** (masked autoencoding for ViT), ViT robustness (occlusion, distribution shift, adversarial patches), and references to DETR and related work.

### [swin-transformer](swin-transformer/)

**Swin Transformer** — Hierarchical vision transformer with **shifted windows**: local window-based self-attention (W-MSA) plus shifted windows (SW-MSA) for cross-window connections. Designed as a general-purpose backbone for detection, segmentation, etc. Notes cover patch merging (56×56→28×28→14×14→7×7), complexity comparison (global MSA vs W-MSA), and the cyclic-shift trick for shifted partitions. No CLS token; operates on patch grids.

### [CLIP](CLIP/)

**Contrastive Language-Image Pre-training** — Joint image and text encoders trained with a contrastive objective on large-scale image–text pairs (e.g., WIT 400M). Enables zero-shot transfer to many vision tasks without dataset-specific training. Notes cover the contrastive objective, prompt templates (e.g., “A photo of a {label}”), training scale (32 epochs, large batch, ViT-L/14@336px), and references to StyleCLIP, CLIPDraw, and open-vocabulary detection.

### [meta-sam](meta-sam/)

**Segment Anything Model (SAM)** — Experiments and examples using Meta’s SAM for segmentation. Contains `sam_example.py` and a PyTorch prediction notebook (`sam_pytorch_predict.ipynb`) for running and exploring the model.

### [multi-object-tracking](multi-object-tracking/)

**Multi-Object Tracking** — Object detection and tracking pipeline, including a **YOLO + DeepSORT** setup (`yolo-deepsort/`) and scripts such as `track.py` and `yolov8.py` for running YOLOv8-based tracking with a requirements file.

---

## Overview

| Directory            | Focus                          |
|----------------------|---------------------------------|
| BERT                 | Encoder pre-training (NLP)     |
| transformer          | Original seq2seq architecture  |
| vision_transformer   | ViT, MAE, image classification |
| swin-transformer     | Hierarchical ViT backbone      |
| CLIP                 | Image–text contrastive learning|
| meta-sam             | Segment Anything examples      |
| multi-object-tracking| YOLO + DeepSORT tracking        |
