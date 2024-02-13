# Original Transformer Implementation to Solve Sequence-to-Sequence Translation.

[Original paper - "Attention is all you need"](https://arxiv.org/abs/1706.03762)

## Input Embeddings 

### What is Input embeddings?
- At the beginning of the Transformer model, there is an embedding layer. This layer is essentially a matrix where each row corresponds
to the embedding vector of a specific token in the vocabulary.
- Each word in the input sequence is represented by its corresponding row in the embedding matrix.

### Word indexing
- Each word in the vocabulary is assigned a unique index.

### Learning Embeddings
- During training, the parameters (weights) of the embedding layer are learned through backpropagation and gradient descent.
- The model adjusts the values in the embedding matrix to minimize the difference between the predicted output and the actual target.

### Word Embedding Initialization
- Can be initialized randomly or using pre-trained embeddings.
- Pre-trained embeddings (such as Word2Vec, GloVe, or FastText) are often used.
- Fine-tuning may occur during training.

### Embedding Dimensionality
- Common dimensions include 50, 100, 200, or 300.

### Code Examples
```text
embedding_layer = nn.Embedding(vocab_size, embedding_dim)
```
## Positional Encoding 

### Reason to have PE
- The Transformer model, doesn't have any inherent sense of the order or position of words in a sequence.
- Self-attention mechanism, while powerful, is permutation-invariant.
- Positional encodings are added to the input embeddings.
- The use of sin/cos function is a clever way to achieve this without introducing additional learned parameters.
- Enable the model to perform effectively on tasks that require understanding the sequential nature of the data.

### Code examples
```
x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
```
-no need to take gradient, particular tensor that no need to learn

## Layer Normalization

### Why use layer of normalization
- The distribution of activations in a layer changes as the model parameters are updated during training, making
it challenging for the model to converge.
- Help to stablize the training of DNN by reducing the internal covariate shift.

### Normalization along Feature Dimension
- LN is applied independently to each feature (or channel) along the last dimension of the input tensor.

### Learnable Parameters
- Layer normalization introduces learnable scale (γ) and shift (β) parameters for each feature.

### Code samples
```
    alpha = nn.Parameter(torch.ones(features))    # alpha is a learnable parameter (scale)
    bias =  nn.Parameter(torch.zeros(features))   # bias is a learnable parameter (shift)
```
## Feed Forward 

### What is the functioinality and why it is applied after self-atttention layer
    
- Complex Representations - The output of self-attention alone may lack the ability to model more intricate, non-linear transformations
FF block provides the model with the capacity to learn and represent non-linear functions of the input data.
- Position-wise Information - For NLP task, position-wise information may be crucial.
FF block helps the model incorporate position-specific transformations.
- Capacity to Model Diverse Patterns
- Adaptability to Task Complexity
- Enhancing Model Depth - Deeper architectures can capture more hierarchical and abstract features, enabling the model to
            learn intricate patterns and representations.

### How Feed Forward block is applied
```
FFN(x) = LayerNorm(x + relu(linear(x) * W1 + b1) * W2 + b2
```
-Linear is the linear transformation
-W1, b1, W2, b2 are learnable parameters
-LayerNorm denotes layer normalization

### Hidden dimension
- The hidden dimension of the FF layer is a hyperparameter that determines the size of the intermediate representations.
- few hundreds to a couple of thousand units
   
### Input and Output Dimensions
- Determined by the dimensionality of the input embeddings.
- Desired output dimensionality of the model.

### Code samples
```
        linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        dropout = nn.Dropout(dropout)
        linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

        linear_2(dropout(torch.relu(linear_1(x))))
```
