# BERT - Bidirectional Encoder Representations from Transformer

[2018 Paper on Bert](https://arxiv.org/abs/1810.04805)

## What is Bert for?

- Bert is for pre-training Transformer's encoder
- Predict masked word
- Predict next sentence

## Original Bert Paper Walkthrough

[Paper reading lists](https://github.com/mli/paper-reading)  
[Bert walkthrough](https://www.youtube.com/watch?v=GDN649X_acE)  

Two steps in Bert frameworks
- pretraining (unlabeled Sentences)
- fine-tuning (supervised, downstream task)


### Bert - Number of parameters
- Base H = 786, L = 12, 30000(word embedding) * H + L * H*H *12 = 110M
- Large H = 1024, L = 24, 30000(word embedding) * H + L * H*H *12 = 340M


## Lecture of Bert

[BERT for pretraining Transformers](https://www.youtube.com/watch?v=EOmd5sUUA_A)

### Data of Bert
- Bert does not need manually labeled data
- Use large-scale data, e.g., English Wikipedia (2.5 billion words)
- Randomly mask 15% words (with some tricks)
- 50% of the next sentences are real

### Training of Bert
- Bert base, 110M parameters, 16 TPUs, 4 days of training (without hyper-parameter tuning)
- Bert large, 235M parameters, 64 TPUs, 4 days of training (without hyper tunning)
