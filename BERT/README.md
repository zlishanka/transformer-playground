# BERT - Bidirectional Encoder Representations from Transformer

[2018 Paper on Bert](https://arxiv.org/abs/1810.04805)

## What is Bert for?

- Bert is for pre-training Transformer's encoder
- Predict masked word
- Predict next sentence

## Tutorial of Bert

[BERT for pretraining Transformers](https://www.youtube.com/watch?v=EOmd5sUUA_A)

### Data of Bert
- Bert does not need manually labeled data
- Use large-scale data, e.g., English Wikipedia (2.5 billion words)
- Randomly mask 15% words (with some tricks)
- 50% of the next sentences are real

### Training of Bert
- Bert base, 110M parameters, 16 TPUs, 4 days of training (without hyper-parameter tuning)
- Bert large, 235M parameters, 64 TPUs, 4 days of training (without hyper tunning)
