# Vision Transformer 

[Original paper - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) 

- There is NO CNN reliance. Previously, a lot of interests focus on combining CNNs with forms of self-attention. 
- A pure transformer applied to image patches can perform very well on iamge classification.
- Unlike NLP task, Pre-training of ViT is supervised.
- It demonstrates that large scale pre-training makes vanilla transformers competitive with STOA of CNNs.
- Similar work done by Cordonnier et al. (2020) use much smaller patch (2x2) because of smaller 32x32 image resolution. (ViT uses 224x224)
- For mid-sized datasets like ImageNet, ViT lags ResNets few percentage points on accuracy due to so-called `inductive biases`
  - CNN has property of Locality and translation equivariance (translation, convolution are interchangable), so don't need large dataset for pre-training
- For large-scale datasets(14M-300M images), ViT attains excellent results on pre-training. ImageNet-21K, JFT-300M
- ViT takes less TPU days during pre-training compared to CNN
- Globally, ViT model attends to image regions that are semantically relevant for classification.
- Future or related works
  - Transformers on other vision tasks, like detection, segmentation
  - Explore self-supervised pre-training

- Implementation details
    - Add CLS token, similar to BERT [class] token, is a learnable embeddings, its output serves as the entire image representation.
    - for 224x224x3 RGB image
       - feature dim: 16x16x3 = 768
       - Tokens num: 224*224/(16*16) = 196 patches
       - full connection layer dim: 768x768, (constant latent vector) to get the patched embeddings, project a vision to NLP
    - Total sequence length: 196+1(cls) = 197
    - Position embeddings: 197x768
    - Multi-head attention layer 
      - if using 12 heads, the single dim of each layer is 768/12 = 64, so each `q, k, v` vector is 197x64
    - Norm layer maintains 197x768
    - MLP layer extends to 197x768x4 first and project it back to 197x768. Basically no change, still `sequence length x feature dim`
    - The output of class token is then transformed into a class prediction via a small MLP with tanh as non-linearity in the single hidden layer.

[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

- MAE to ViT is like BERT to Transformer. Self-supervised!
- Audoencoder means more like auto-regression, label and sample come from the same source (in NLP, use the first couple of words to predict next word), predict/reconstruct masked image itself.
- Two core designs
  - Asymmetric encoder-decoder architecture, encoder operates only on the visible subset of patches, along with lightweight decoder that reconstruct the mask tokens.
  - Masking a high proportion of the input image, i.e., 75%, yields a nontrivial and meaningful self-supervisory task.

- Since encoder is only dealing with 25% of unmasked input patches, it has less computational complexity
- 

[Intriguing Properties of Vision Transformers](https://arxiv.org/abs/2105.10497)

- ViT shows better handling vision tasks in situations like `Occlusion`, `Distribution Shift`, `Adversarial Patch`, `Permutation` as compared to CNN.
- ViT is robust on shape and texture analysis.
- ViT is robust on Adversarial and Natural Perturbations.

[Residual Connection paper - "Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385)

## References 

[Lucidrains](https://github.com/lucidrains/vit-pytorch) 

[Transformers for Vision from D2L Tutorial](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)

[ViT Blogpost by Francesco Zuppichini](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632) 

[Medium Blogpost by Brian Pulfer](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c)

## Lecture on ViT

[Vision Transformer for Image Classification](https://www.youtube.com/watch?v=HZ4j_U3FC94)

## Related work

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
