import torch
import torch.nn as nn
import math

# Overview of the data flow (encoder-decoder, as in "Attention Is All You Need"):
#
#   source tokens --> InputEmbeddings --> PositionalEncoding --> Encoder (N blocks) --+
#                                                                                     | encoder_output
#   target tokens --> InputEmbeddings --> PositionalEncoding --> Decoder (N blocks) <-+
#                                                                    |
#                                                              ProjectionLayer --> log-probabilities over target vocab
#
# Tensor shapes used in the comments below:
#   Batch    = number of independent sentences processed together
#   seq_len  = number of token positions per sentence (the "context window", padded to a fixed length)
#   d_model  = width of each token's vector (the "residual stream" width, 512 by default)

class InputEmbeddings(nn.Module):
    """Turns token IDs into d_model-wide vectors (a learned lookup table)."""

    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # mapping: number ---> vector(512)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: (Batch, seq_len) integer token IDs --> (Batch, seq_len, d_model)
        # The paper scales the embeddings by sqrt(d_model) so they are not drowned out
        # by the positional encoding that is added right after.
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """Adds a fixed (not learned) position signal, because attention alone has no sense of word order."""

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)) # d_model/2
        # Apply the sin to even position
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply the cos to odd position
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        pe = pe.unsqueeze(0) # (1, seq_len, d_model), the leading 1 broadcasts over the batch

        # A buffer is saved with the model and moved to the GPU/MPS with it, but is not a trainable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, seq_len, d_model). Position i of the sentence gets row i of pe added to its vector.
        # Slicing to x.shape[1] lets sentences shorter than the maximum seq_len work too.
        # require_-grad(False), make this particular tensor not learned
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class  LayerNormalization(nn.Module):
    """Rescales each token's vector to mean 0 and std 1 across its d_model numbers, then applies a learned scale and shift."""

    def __init__(self, features:int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))    # Multiplied, alpha is a learnable parameter
        self.bias =  nn.Parameter(torch.zeros(features))    # Added, bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim=True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim=True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean)/(std + self.eps) + self.bias

# Feed Forward Block (the "MLP" part of each layer). Applied to every token position independently,
# so it does not move information between tokens; that is attention's job.
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention. All h heads share one set of weight matrices (w_q, w_k, w_v);
    the heads are made in forward() by reshaping the d_model outputs into h slices of d_k numbers."""

    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h      # Number of heads
        assert d_model % h == 0 , "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        # Each of these is one big matrix covering all heads side by side (not one Linear per head)
        self.w_q = nn.Linear(d_model, d_model, bias=False)  #Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  #Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  #Wv

        # Mixes the concatenated head outputs and writes the result back to d_model
        self.w_o = nn.Linear(d_model,d_model, bias=False)   #Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # Scaled dot-product attention, run for all batches and heads at once.
        d_k = query.shape[-1]

        # Compare every query with every key: score[i][j] = how much position i looks at position j.
        # Dividing by sqrt(d_k) keeps the scores from growing with head size, which would make softmax too sharp.
        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            # Where mask is 0 (padding, or future tokens in the decoder) set the score to a huge negative
            # number, so softmax turns it into a weight of about 0.
            attention_scores.masked_fill_(mask == 0, -1e9)
        # Each row becomes weights that sum to 1 over the positions being looked at
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Weighted average of the value vectors, using those weights
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Self-attention passes the same tensor as q, k and v. Cross-attention passes the decoder
        # stream as q and the encoder output as k and v.
        # Step 1: project the inputs with the shared weight matrices
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)   # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # Step 2: split d_model into h heads of d_k numbers, then move the head dimension next to Batch
        # so that attention treats each head separately.
        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # Step 3: attention inside each head. Scores are kept in self.attention_scores for visualization.
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Step 4: put the heads back side by side (concatenate). contiguous() is needed before view()
        # because transpose() leaves the tensor non-contiguous in memory.
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Step 5: output projection mixes information across heads
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """Wraps a sublayer (attention or feed-forward) with normalization and a skip connection.
    This is the "pre-norm" layout: normalize first, run the sublayer, add the result back to the input.
    (The original paper normalizes after the addition instead.)"""

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # sublayer is a function, so the same wrapper works for attention and feed-forward.
        # The "+ x" is the residual stream: the sublayer adds to it instead of replacing it.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """One encoder layer: self-attention, then feed-forward, each wrapped in a residual connection."""

    def __init__(self, features:int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Self-attention: q, k and v are all x. Source tokens can see each other in both directions;
        # src_mask only hides padding.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    """A stack of N encoder blocks. The final LayerNormalization is needed because the blocks are pre-norm."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    """One decoder layer: masked self-attention, cross-attention to the encoder output, then feed-forward."""

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
       super().__init__()
       self.self_attention_block = self_attention_block
       self.cross_attention_block = cross_attention_block
       self.feed_forward_block = feed_forward_block
       self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked self-attention over the target so far. tgt_mask hides padding and future positions,
        # so each target token only sees earlier ones (causal attention).
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        # Cross-attention: query comes from the decoder, so "what am I looking for" is a target-side question.
        # key, value come from encoder output, use src_mask as well
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    """A stack of N decoder blocks. Every block receives the same encoder_output."""

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """Final step: turns each position's d_model vector into a score for every word in the target vocabulary."""

    def __init__(self, d_model:int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Output is log-probabilities of the next target token at each position.
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    """Holds all the pieces. forward() is not defined: callers run encode(), decode() and project() separately,
    so that at inference time the source can be encoded once and decode() repeated per generated token."""

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # token IDs --> embeddings --> add positions --> encoder stack
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # same steps for the target side, plus cross-attention to encoder_output inside the decoder
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, tgt_vocab_size)
        return self.projection_layer(x)

# build transformer based on given parameters
# Defaults are the base model of the original paper: d_model=512, N=6 layers, h=8 heads, d_ff=2048.
# N = number of encoder blocks and also number of decoder blocks.
# d_ff = inner width of the feed-forward block.
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int,
                      tgt_seq_len:int, d_model:int = 512, N:int = 6, h:int = 8,
                      dropout:float = 0.1, d_ff:int = 2048) -> Transformer:
    # Create emmbeding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positioinal encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks. Each block gets its own freshly created attention and
    # feed-forward modules, so the N layers have separate weights.
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks (each has two attention blocks: self-attention and cross-attention)
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed,
                              src_pos, tgt_pos, projection_layer)

    # Initialize the parameters: Xavier-uniform for every weight matrix (2+ dimensions),
    # while 1-D parameters (biases, LayerNorm alpha/bias) keep their defaults.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
