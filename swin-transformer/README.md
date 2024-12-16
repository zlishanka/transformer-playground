# Swin Transformer

[Original paper: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

General-purpose backbone for computer vision. It can replace CNN on many downstream tasks like detection, segmentation,etc.

## Approach 

- Input image 224x224x3 ---> patch size 4x4, 56x56x48 (4x4x3)
- patches ---> linear embedding  ---> 56x56x96  ---> 3136x96 (swin Transformer block) 
- Patch Merging 56x56x96 ---> STB ---> 28x28x192 ---> STB ---> 14x14x384 ---> STB ---> 7x7x768
- Swin Transformer doesn't use CLS token

### Shifted Window based self-attention

- Standard Transformer for image classification conduct a global self-attention
- Global computation leads to quadratic complexity with respect to the number of tokens, so not suitable for dense prediction or representing high-resolution image
- Sub-divide a non-overlapping 56x56 window further into  MxM patches (say M=7), W-MSA much less computation as compared to MSA
```
Ω(MSA) = 4hwC2 + 2(hw)2C, (1)
Ω(W-MSA) = 4hwC2 + 2M2hwC, (2)
```
- Shifted window partitioning in successive blocks to enhance cross-window connections, W-MSA first followed by SW-MSA
- Cyclic shift to solve shifted windows' size disparity
