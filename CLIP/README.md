# CLIP - Contrastive Language-Image Pre-training  

OpenAI CLIP (Contrastive Language-Image Pre-training) is a groundbreaking model developed by OpenAI that bridges the gap between computer vision and natural language processing.  
CLIP is a technique for training a pair of neural network models, one for image understanding and one for text understanding, using a contrastive objective.

CLIP model transfers non-trivially to most tasks and is often competitive with a fully supervised basline without the need for any dataset specific training.

## OpenAI CLIP

[Original paper: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

- For N image-text pairs, use text-encoder and image-encoder, we get N text and image features,
- CLIP jointly trains an image encoder and a text encoder to predict correct pairings
- OpenAI collects a image-text dataset that include 400M samples, high quality and clean
- Create dataset classifier from label text (1000 categories are from existing dataset, say ImageNet)
    - Prompt template, `A photo of a (object)`, for 1000 text prompts, through text encoding we get 1000 text features
 
- A crucial difference between weakly supervised models and recent explorations of learning image representations from natural language is `scale`.
- Benchmark the zero-shot transfer performance of CLIP on over 30 existing datasets.
- Approach
    - Natural Language Supervision
        - Early work wrestled with the complexity of natural language when using topic model and n-gram representations
        - improvements in deep contextual representation learning give much better/effective tool
        - Previous image-text pair datasets - MS-COCO, Visual Genome(too small), YFCC100M(low quality labeling though)
        - Create 400M dataset called WIT for WebImageText.
        - Initial approach, similar to VirTex, jointly trained an image CNN and text transformer to predict the caption of an image
        - Contrast learning - to only judge if a given image-text pair can match to each other.
        - Swap the predictive objective for a constractive objective, training efficiency improves 400%
- Training
    - train all models for 32 epochs, use Adam optimizer, minibatch size of 32768
    - [Reference: How to Train Really Large Models on Many GPUs?](https://lilianweng.github.io/posts/2021-09-25-train-large/)
    - choose final model to be `ViT-L/14@336px`

- Experiments
    - zero-shot transfer:
    - Prompt engineering and ensembling
        - use a sentence not single word during pre-training
        - when doing inference, there would a distribution gap
        - use `prompt template` to solve above problem, like `A photo of a {label}.`
        - CLIP model uses about 80 prompt templates, check github openai/CLIP, `imagenet_templates`
     
    - Representation learning
    - Comparison to Human Performance
        
## CLIP realted papers

[StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://arxiv.org/abs/2103.17249)

[CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders](https://arxiv.org/abs/2106.14843)

[Open-vocabulary Object Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921)
