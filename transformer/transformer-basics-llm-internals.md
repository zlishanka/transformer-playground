# Transformer Basics: LLM Internals for Model Diffing

Split out of [[delta-crosscoder-summary-arxiv-2603.04426]] on 2026-11-09, where it was background for the Part 2 glossary (activations, SAEs, crosscoders, steering). "The paper" below means Delta-Crosscoder (arXiv 2603.04426). See also [[model-diffing-for-fine-tuning-references]] and [[finetuning-bias-guardrails-and-tests-arxiv-2510.13900]].


All of this is general background knowledge, not from the paper. Model figures (Llama 3.1 8B: 32 layers, hidden size 4,096, 32 attention heads, MLP inner width 14,336) are from memory and have not been checked against the model's config.

## Decoder-only models and "middle layers"

- **The original transformer (2017)** had an encoder (reads the input) and a decoder (generates the output while attending to the encoder). T5 and the original translation models follow that design.
- **Modern chat LLMs** (Llama, Qwen, Gemma, the GPT models) keep only the decoder part: one stack of identical blocks (32 in Llama 3.1 8B), each an attention-plus-MLP step on the residual stream, with causal attention (a token sees only earlier tokens). Everything in this glossary assumes that kind of model.
- **BERT-style models** are encoder-only. They are not used for chat or generation and do not appear in these papers.
- **So "middle layers" means the middle of that single stack,** for example around layers 12-20 of 32. Early layers sit near the token embeddings and late layers near the output logits. "Middle" is a rough label for depth, not an official boundary.
- That the models in the papers (Llama, Qwen, Gemma) are decoder-only is from background knowledge, not checked against each model card.
- For an encoder-decoder model, "middle" would be ambiguous. Papers then usually give a specific layer number and say which stack it belongs to, which is worth checking.

**When an encoder-decoder design is used instead** (background knowledge, not from the papers). It fits when the task turns a complete input into a different output, and the input is worth reading in full before generating anything.

- **The encoder reads the whole input at once, in both directions.** Every input token can see every other input token, including later ones. A decoder-only model can't, because its attention only looks backward.
- **The decoder generates the output one token at a time,** attending to the encoder's finished representation of the input (cross-attention), so it has a full, stable view of the source while writing.
- **The input is encoded once,** which is efficient when the input is long and the output short, or the reverse.
- **Typical uses:**
  - **Translation:** the original transformer's task. The source goes into the encoder and the target comes out of the decoder.
  - **Summarization and text-to-text tasks:** T5 and BART are encoder-decoder models.
  - **Speech recognition:** Whisper encodes the audio and decodes the text.
  - **Vision-language and image captioning:** an image or audio encoder feeds a text decoder. Many multimodal systems keep a separate encoder for the non-text input and attach it to a decoder-only language model.
- **Why decoder-only won for chat models:**
  - **Simpler:** one stack handles everything, with instructions and answers in the same stream of text.
  - **Scales well:** it trains on huge amounts of plain text by predicting the next token, with no input-output pairs needed.
  - **Flexible:** at scale it handles translation, summarization and the rest from the prompt alone, so the dedicated structure matters less.
- Encoder-decoder models are still used where input and output are clearly different things, such as the speech and vision cases above.

## Hidden size and "the embedding"

- **4,096 is the model's hidden size** (`d_model`). The model's own first step turns each token ID into a vector of that width, and every layer keeps that width. It is not a separate "embedding model."
- A standalone embedding model (used for search or RAG, one vector per text) is a different thing, with its own unrelated output size.
- Some internal pieces are wider than 4,096 (the MLP's inner layer). Activation methods in these papers usually read the 4,096-wide stream, but I have not checked which activations Delta-Crosscoder uses.

## Tracing a token's vector through the layers

You need an open model so you have the weights. Untested sketch using Hugging Face `transformers`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "meta-llama/Llama-3.1-8B"          # any open model you can load
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)

prompt = "US capital city is Washington"
enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)

# find which token(s) cover the word "Washington"
start = prompt.index("Washington")
end = start + len("Washington")
offsets = enc.pop("offset_mapping")[0]
positions = [i for i, (s, e) in enumerate(offsets.tolist()) if s < end and e > start]

with torch.no_grad():
    out = model(**enc, output_hidden_states=True)

# out.hidden_states: tuple of (num_layers + 1) tensors, each [1, seq_len, 4096]
for layer, h in enumerate(out.hidden_states):
    vec = h[0, positions[-1]]             # the 4096-number vector for "Washington" at this layer
    print(layer, vec.shape)
```

- **Tokenization:** A word may split into several tokens. The offset mapping finds the token positions covering it. Using the last piece is a choice.
- **`output_hidden_states=True`:** Returns one tensor per layer, shape `[batch, tokens, 4096]`. The first is the embedding step's output.
- **A token only sees what came before it** (causal attention), so its vector cannot reflect later words.
- **A raw vector is not readable.** Compare it to something: the same position in another prompt, the base vs fine-tuned model, or a projection onto a steering direction or SAE latent. The "logit lens" projects a layer's vector to the vocabulary to see which word the model would predict at that point.
- **Finer values need hooks** (PyTorch forward hooks, or a library like TransformerLens) to get attention or MLP outputs inside a layer.
- From memory, the last returned state in Hugging Face is, I believe, after the final normalization step. Not verified.

## Which numbers or combinations to trace (directions)

- **You don't pick individual numbers.** In a transformer the 4,096 numbers are not 4,096 separate nodes each with a job. One number responds to many unrelated things, and one concept is spread across many numbers. Per-number tracing mostly gives noise.
- **A concept lives in a direction,** a weighted combination of all the numbers. Think of the 4,096 numbers as a point in a 4,096-dimensional space. A direction is a line through that space, and you read a concept by asking how far the point lies along it (a dot product). The axes are arbitrary: any rotated axes describe the same space.
- **Ways to find a direction:**
  - **Difference of means:** average vectors from prompts with and without the property, then subtract (how steering vectors are made).
  - **Linear probe:** train a simple classifier on the vectors, such as "true vs false capital." Its weights are a direction.
  - **SAE or crosscoder latents:** learn many directions at once. Delta-Crosscoder does this for the activation difference between two models.
  - **PCA on the differences:** find the directions where base and fine-tuned vectors differ most.
- **Testing whether a direction matters** (finding one only shows correlation):
  - **Steering:** add or subtract the direction and see whether the output changes.
  - **Activation patching:** copy the vector from one run (say the "Paris" position in "France capital city is Paris") into another (the "Washington" position in "US capital city is Washington") at one layer and see whether the output changes. This shows which layers and positions carry the fact.
- **Which layer:** Sweep across layers and see where the signal is clearest. Concept-level information is often strongest in middle layers, but that is a general pattern, not confirmed for any specific model.
- **Sketch for a false-fact check (my suggestion, not from a paper):** collect true and false statements, get the final-token vector at every layer, compute a difference of means and train a probe per layer, then test by steering or patching. For a fine-tuned model, repeat on both models and compare.
- **Superposition:** A 4,096-dimensional space has only 4,096 perfectly independent directions, but the model needs far more concepts. It uses nearly independent directions, so thousands of concepts overlap with small interference. This is why SAEs and crosscoders exist.

## The residual stream

- **What it is:** The 4,096-number vector each token carries through the whole model. Each layer reads it, computes something, and **adds** the result back instead of replacing it: `new stream = old stream + layer's output`. The "residual" is this add-back step (a skip connection).
- **Analogy:** A notebook passed down a line of workers. Each reads it, adds notes at the end, and passes it on.
- **Consequences:**
  - Each token position has its own stream. Streams only exchange information through attention.
  - The "activation at layer L" is a snapshot of the stream after layer L, which is what `output_hidden_states=True` returns.
  - Early information persists unless something cancels it, which is why concepts can be read and steered at a chosen layer.
  - Layers communicate only through the stream, with no direct wire between layer 5 and layer 20.
- **Reading and writing:** A block writes by adding a vector, pushing the stream along some directions. It reads by multiplying the stream by a weight matrix, which effectively picks out how much of certain directions is present.
- **Normalization (from memory):** Each block normally rescales the stream before reading it (RMSNorm in Llama). The stream itself keeps the plain sum. Not checked against a specific model's code.
- **Stream width and token count stay fixed across layers** (background knowledge; Llama figures from memory).
  - **Width:** Every token's stream stays 4,096 numbers at every layer. The add-back step only works if the layer's output has the same width as the stream, so `W_O` and the MLP's last matrix both end at 4,096.
  - **Inner pieces are scratch space:** Q is 4,096, K and V are 1,024, and the MLP middle is 14,336. All return to 4,096 before the next layer.
  - **Width changes only at the ends:** The embedding step turns a token ID into 4,096 numbers, and the output layer turns the final vector into about 128k vocabulary scores.
  - **Token count:** A 14-token prompt has 14 vectors going into layer 1 and 14 coming out of every layer. Layers don't drop, merge or add tokens. Attention only moves information between them.
  - **Apparent exceptions:** During decode only the new token's vector is computed per layer, but the KV cache still holds a K and V row for every earlier token at that layer. After the last layer only the final position is needed to pick the next token.
  - **Not covered here:** Funnel-style or CNN-like designs shrink the number of positions in deeper layers. I don't know of standard chat LLMs doing this, and I haven't checked each model.
- **Q, K and V are projections of the stream, not the embedding** (background knowledge). The embedding is only the stream's starting value. At every layer, Q, K and V are the normalized stream times that layer's `W_Q`, `W_K` and `W_V`, so each layer gives a token different ones, and they reflect the context absorbed so far. Only layer 1 starts from the embedding itself. Q is used once and dropped, while K and V are cached.
- **Transformer vs plain MLP:** A transformer does contain MLPs (each layer has an attention part and an MLP part). The difference is that all parts read from and write to one shared stream, instead of passing activations through a simple chain of layers. The MLP's inner units are the closest thing to classic MLP nodes. Many studies suggest factual associations are stored mostly in these MLPs. That is a general finding I recalled, not verified here.

## Attention: how information moves between tokens

Each token's stream starts out mostly knowing its own token. Attention lets a token read from earlier tokens' streams and copy useful information into its own. For "US capital city is Washington," when processing "Washington":

1. **Query:** "Washington" builds a query vector from its own stream, roughly "what am I looking for?"
2. **Keys:** Each earlier token ("US", "capital", "city", "is") has a key vector, roughly "what I'm about." The query is compared with each key by a dot product. A high score means relevant.
3. **Values:** Each earlier token also has a value vector, what it offers to share. "Washington" takes a weighted average of those values, with the scores turned into weights that sum to 1.

The result is added to the "Washington" stream, following the residual-stream rule (attention writes, it doesn't overwrite).

```
 "US"      "capital"    "city"      "is"     "Washington"
  │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼
 keys/values keys/values keys/values keys/values  query
  └───────────┴───── compare, weight, copy ───────→ added into "Washington" stream
```

- **Only backward:** A token attends only to itself and earlier tokens (a causal mask).
- **Many heads in parallel:** Each layer has many heads, each with its own query, key and value rules, so different heads can specialize. Their outputs are combined and added to the stream.
- **Two separate decisions:** The query-key part decides where to look. The value-and-output part (OV) decides what to move and which directions to write.
- **Facts (general finding, not verified here):** For "capital of X," attention moves information about the subject into the stream where the answer forms, and the MLP step often does the lookup.
- **Link to the thread:** The abstract of 2604.08524 reports that steering vectors act mainly through the OV pathway, not the QK pathway. Only the abstract was read.

**Attention weight sizes** (background knowledge; the Llama 3.1 8B figures are from memory and unverified).

- **Sizes depend on the token's vector dimension (hidden size) and the head design, not on the context window.** A weight matrix multiplies one token's vector, which has 4,096 numbers, so each matrix must accept 4,096 inputs. The number of tokens varies per prompt, so it cannot set a weight shape. The same weights are applied to every token position.
- **Llama 3.1 8B:**
  - **Hidden size:** 4,096.
  - **Query weights (W_Q):** 4,096 × 4,096, which is 32 heads × 128 numbers per head.
  - **Key and value weights (W_K, W_V):** 4,096 × 1,024 each. Llama 3 uses grouped-query attention, where many query heads share fewer key/value heads (here 8 × 128).
  - **Output weights (W_O):** 4,096 × 4,096. They combine the heads' results and write them back into the stream.
  - **Parameters in one attention block:** roughly 42 million (2 × 4,096² for Q and O, plus 2 × 4,096 × 1,024 for K and V).
- In a plain multi-head design without grouped-query attention, K and V would also be 4,096 × 4,096. The designer chooses the number of heads and their width. The input side is always the hidden size.
- **What the context window does affect (not the weights):**
  - **Attention scores:** a token-by-token table that grows with the square of the prompt length. This is activation memory during a run.
  - **KV cache:** during generation, the model stores earlier tokens' keys and values, and this memory grows with context length.
  - **Position information:** many modern models, Llama included, use rotary position embeddings, which have no learned weights. This is from memory for Llama.
- A 128k-token model and an 8k-token model can have identically shaped Q, K and V weights. The window length limits what the position scheme and memory budget can handle.

## What W_Q, W_K and W_V do (projections)

Background knowledge, plus the code reading from `model.py` in this directory (`MultiHeadAttentionBlock`: `d_model` 512, 8 heads of 64, from `config.py` and the `build_transformer` defaults). The Llama figures are from memory and unverified.

- **"Project" means multiply by a learned matrix.** `W_Q`, `W_K` and `W_V` each take a token's stream vector and re-express it for one job in attention.
- **Why the stream is not used directly:** After embedding, a token has one vector, but attention needs it for three purposes.
  - **Q (`w_q`):** "what am I looking for?"
  - **K (`w_k`):** "what do I contain, for others to match against?"
  - **V (`w_v`):** "what content do I hand over if someone attends to me?"
- **What the three matrices add:**
  - Without them, the score between two tokens would be `x_i · x_j`. That is symmetric and only measures how similar the vectors are.
  - With them, the score is `(x_i W_Q) · (x_j W_K)`. The model can learn a relation such as "a verb looks for its subject", which is not the same as "these two tokens are similar".
  - Matching (Q·K) and content (V) are separate, so what makes a token easy to find need not equal what it passes on.
- **What each one does in the code:**
  ```python
  query = self.w_q(q)   # (Batch, seq, 512) -> (Batch, seq, 512)
  key   = self.w_k(k)
  value = self.w_v(v)
  ```
  - **Q and K** are multiplied to get the scores (`query @ key.T`), and softmax turns the scores into weights. They only decide where to look.
  - **V** is then averaged using those weights (`attention_scores @ value`). It is the only one of the three whose content flows into the output.
  - **`W_O` (`w_o`)** is a fourth matrix, not a Q, K or V. It mixes the concatenated head outputs and writes the result back into the stream.
- **Each matrix covers all heads side by side.** In `model.py`, `W_Q` is one 512 × 512 matrix. Columns 0-63 give head 0's queries, columns 64-127 give head 1's, and so on, so it acts as eight 512 × 64 projections. Each head gets its own learned view of the same token, which lets heads specialize. This is also why the code looks like a single head: the split happens by reshaping in `forward`, not by separate modules.
- **Same weights for every position, separate weights per layer.** One `W_Q` is applied to all tokens in a layer. Each of the 6 encoder layers, and each attention block in the decoder, has its own set.
- **Self-attention vs cross-attention:** The matrices have the same shape in both. Only the input differs.
  - **Self-attention:** `W_Q`, `W_K` and `W_V` all read the same stream.
  - **Cross-attention (encoder-decoder models like this one):** `W_Q` reads the decoder stream, while `W_K` and `W_V` read the encoder output. The decoder's question is matched against the source tokens, and the source content is copied back.
- **These are the learned part of attention.** The dot product, softmax and weighted average have no parameters. In the terms used above, `W_Q` and `W_K` set where to look (the QK pathway), and `W_V` and `W_O` set what to move and where to write it (the OV pathway).
- **Difference from Llama:** `model.py` uses plain multi-head attention, so `W_K` and `W_V` are full 512 × 512 (8 K/V heads for 8 Q heads). Llama 3.1 8B uses grouped-query attention, with smaller `W_K` and `W_V` as described earlier.

## Multi-head attention: meaning and purpose

Background knowledge, plus the code reading from `model.py` (8 heads of 64 numbers, `d_model` 512). The findings about what heads learn are general recollections and have not been checked against this model.

- **Meaning:** Multi-head attention runs several smaller attention operations in parallel on the same tokens, then combines their results. Each head is a complete attention: its own Q, K and V views, its own scores and its own softmax.
- **The problem it solves:**
  - A single head produces one weight distribution per token, which is one answer to "where should this token look?" and one blend of values.
  - A token often needs several kinds of information at once. In "the animal didn't cross the street because it was tired", "it" needs its referent ("animal"), and other tokens may need the nearby verb, the subject, or a matching bracket.
  - If one softmax has to serve all of these, the weights blur together.
- **What multiple heads give you:**
  - **Several independent looks per token:** Each head has its own `W_Q` and `W_K` columns, so it can learn its own notion of relevance and its own attention pattern.
  - **Separate content per look:** Each head has its own `W_V` slice, so it can pass on different information, not only attend to different places.
  - **Roughly the same cost as one big head:** With `d_k = d_model / h`, 8 heads of 64 cost about the same compute and parameters as one head of 512.
- **The five steps in `forward`:**
  1. Project the input with `W_Q`, `W_K` and `W_V`.
  2. Split the 512 numbers into 8 slices of 64, one per head.
  3. Run attention inside each head. Heads never see each other's slices here.
  4. Concatenate the 8 outputs back into 512 numbers.
  5. Apply `W_O`, which mixes information across heads and writes the result into the residual stream. This is where the heads first combine. Without it, the heads would stay as isolated lanes.
- **Caveats on "specialization":**
  - Heads are not assigned jobs by design. Nothing in the code tells head 3 to track syntax. Training only gives them the freedom to differ, and some settle into recognizable roles.
  - Interpretability work reports specialized heads, such as heads that copy the previous token or match repeated patterns. It also reports that many heads are redundant and can be removed with little loss.
  - The safe summary is that multiple heads allow several attention patterns per layer. They do not guarantee that each one does something distinct.
- **Cost of the design:** Scores are computed per head, so the score tensor is `(Batch, h, seq, seq)`. More heads means a larger score tensor for the same sequence length, on top of the quadratic growth with context length described above.
- **Variants:** Llama keeps many Q heads but shares K and V across groups of them (grouped-query attention). This keeps most of the multi-head benefit while shrinking the KV cache.

## What sets a model's parameter count

Background knowledge; all Llama 3.1 8B figures are from memory and have not been checked against the model's config.

- **Five factors:**
  1. **Hidden size** (4,096). Almost every matrix has it as one side.
  2. **Number of layers** (32). The per-layer weights repeat this many times.
  3. **Attention design:** number of heads, key/value heads and head width. This fixes the Q, K, V and output sizes.
  4. **MLP inner width** (14,336, the wider internal layer). This is the biggest piece.
  5. **Vocabulary size** (about 128k). It sets the embedding table (vocabulary × hidden size) and the output layer, which is the same size if not shared.
- **Check with Llama 3.1 8B:**
  - **Attention per layer:** about 42M, as computed above.
  - **MLP per layer:** three matrices of 4,096 × 14,336 (about 59M each), so about 176M. Llama's MLP uses three matrices (gate, up, down).
  - **One layer:** about 218M. Times 32 layers is about 6.98B.
  - **Embeddings and output layer:** 128,256 × 4,096 is about 0.53B each, about 1.05B total.
  - **Total:** about 8.03B, which matches the "8B" name and supports the figures.
- **The MLP holds about 80% of each layer's weights,** which is why the idea that facts are stored mostly in MLPs matters (a general finding, not verified here).
- **Does not change the parameter count:**
  - **Context window:** not a factor.
  - **Number precision** (bf16 vs 4-bit): changes the memory the weights take, not how many there are.
- **Exception, mixture-of-experts models:** they hold many copies of the MLP and use only a few per token, so total parameters exceed the parameters used per token.

## Context, the KV cache, and where it lives

Background knowledge, except the `/context` figures, which come from this session. The Llama 3.1 8B figures are from memory and have not been checked against the model's config.

- **The model keeps no memory between turns.** On every turn the client resends the whole conversation as one token sequence: system prompt, rules files, tool and skill descriptions, all earlier messages, tool results, and the new prompt. The answer is generated at the end of that sequence, and each new token attends back over all the earlier tokens.
- **Example from this session** (`/context` output, 2026-11-08, a snapshot):
  - **Total:** 190.5k tokens of a 1M window (19%).
  - **Conversation messages:** 159.8k (about 84%). This includes tool calls and their results, such as fetched pages and files read.
  - **Fixed overhead:** about 31k (about 16%): system prompt 3.9k, system tools 5.2k, MCP tools 0.6k, MCP instructions 2k, custom agents 2.2k, memory files 6.9k, skills 10k.
  - **Deferred tools are not counted** (about 315k if all were loaded). They are only a name list until a schema is loaded with ToolSearch. Loading the one Slack thread reader schema cost 608 tokens.
  - **Autocompact buffer:** 33k reserved for the automatic summarizing step.
- **What the KV cache is:** To produce each new token, the model needs the keys and values of all earlier tokens at every layer. Instead of recomputing them for every new token, it computes them once, stores them, and reuses them. That store is the KV cache. Queries are not cached, since only the current token needs its own.
- **Size:** `2 (K and V) × layers × key/value width × bytes per number × tokens`.
  - **Llama 3.1 8B illustration:** 2 × 32 × 1,024 × 2 bytes (bf16) = 131,072 bytes, about 128 KB per token. For 190.5k tokens that is about 25 GB.
  - **Why it is not larger:** Grouped-query attention uses 8 key/value heads instead of 32, which makes the cache about 4x smaller than full multi-head attention would be (derived from the sizes above).
  - **This is an illustration only.** The model in this session has a different, undisclosed architecture, so its real cache size is unknown.
  - **Scaling:** The cache grows linearly with context length. Long context is mostly a memory cost, not a weight cost.
- **How the cache is used to generate** (background knowledge). The weights are not in the cache. The cache holds only keys and values. The weights are the model's fixed parameters, which stay in GPU memory the whole time.
  - **Two phases:**
    1. **Prefill:** The model processes the whole prompt (all ~190k tokens in this session) in parallel. At every layer it computes each token's key and value and stores them in the cache.
    2. **Decode:** It then generates one new token at a time, reusing the cache.
  - **One decode step for the new token** (repeated at each layer, 32 in Llama 3.1 8B):

    ```
    new token's stream (4096)
       │  multiply by W_Q, W_K, W_V      ← weights, read from GPU memory
       ▼
     q, k, v for this token only
       │  append k, v to this layer's cache
       ▼
     compare q with ALL cached keys  →  weights over earlier tokens
       │  weighted average of ALL cached values
       ▼
     multiply by W_O, add to the stream
       ▼
     MLP weights, add to the stream
       ▼
     next layer …  →  last layer → scores for the next word → pick a token
    ```

    The picked token becomes the new input, and the loop repeats.
  - **Computed fresh:** Only the new token's stream, query, key and value, plus its attention and MLP.
  - **Read from the cache:** The keys and values of all earlier tokens, at that layer.
  - **Not recomputed:** The earlier tokens' own streams. Attention is causal, so an earlier token's stream never changes when later tokens arrive. That makes the cache exact, not an approximation.
  - **Not stored:** Old queries, since only the current token needs its own.
  - **Read from memory per step:** All the weights (about 16 GB for an 8B model in bf16, from 8.03B × 2 bytes) plus the whole KV cache (about 25 GB in the illustration above). Decode is therefore often limited by memory bandwidth, meaning how fast the GPU can read that data, more than by arithmetic. This is general knowledge and has not been verified for the model in this session.
- **Where it lives:**
  - **Hosted model:** On the provider's servers. Claude Code on a laptop only holds the transcript text and resends it each turn.
  - **Local open model:** In your own GPU or RAM.
  - **Storage details** for the hosted case (GPU memory, CPU offload, disk, compression) are not public as far as I know.
- **Prompt caching:** The server can reuse the already-processed beginning of the input if it is identical to what was sent before. The session's tool descriptions state a 1-hour cache lifetime for this session. If the cache has expired, the full text is reprocessed. Caching saves recomputation. It does not change what the model sees.
- **Compaction:** When the history nears the context window, the harness summarizes older parts and sends the summary instead, so the model then sees the summary, not the original text. This session is far from that point (19% of 1M).

## Worked example: Q, K and V for a prompt and an answer

Background knowledge. No model was run, so there are no real Q, K or V values here, only what each is and how it flows. Llama 3.1 8B sizes are from memory. Whole words stand in for tokens to keep it simple.

**Input:** `what is the capital city of US`. **Output:** `capital city of US is washington DC`.

In a decoder-only model, every token gets its own Q, K and V. There is no separate set of "input tokens" and "output tokens" playing different roles. The prompt and the answer sit in one list of 14 tokens, and the answer is added one token at a time:

```
position:  1     2   3    4       5     6   7  │  8       9     10  11  12  13          14
token:     what  is  the  capital city  of  US │  capital city  of  US  is  washington  DC
           └──────────── prompt ──────────────┘  └─────────────── generated ───────────────┘
```

- **What each token has:** Each token's stream is multiplied by the three weight matrices, at every layer.
  - **Q:** 4,096 numbers, split into 32 heads of 128. "What am I looking for?"
  - **K:** 1,024 numbers, 8 heads of 128. "What I'm about."
  - **V:** 1,024 numbers, 8 heads of 128. "What I offer to copy."

**Weight matrices that produce them** (per layer, Llama 3.1 8B sizes from memory). The input side is always the stream vector, 4,096 numbers.

| Matrix | Role | Shape (in × out) | Numbers | Output |
| :----: | :--- | :--------------: | ------: | :----- |
| `W_Q` | Query | 4,096 × 4,096 | ≈ 16.8M | 32 heads × 128 |
| `W_K` | Key | 4,096 × 1,024 | ≈ 4.2M | 8 heads × 128 |
| `W_V` | Value | 4,096 × 1,024 | ≈ 4.2M | 8 heads × 128 |
| `W_O` | Output | 4,096 × 4,096 | ≈ 16.8M | Back into the stream (4,096) |
| **Total** | | | **≈ 41.9M** | |

> [!note] `W_O` is not Q, K or V
> It mixes the heads' attention output back into the stream. The total matches the ~42M per attention block computed earlier.

- **Square `W_Q` is a design choice:** Query size is heads × head size = 32 × 128 = 4,096, which happens to equal the hidden size.
- **`W_K` and `W_V` are smaller** because 8 key/value heads are shared across the 32 query heads (grouped-query attention).
- **Same matrix for every token:** The matrices do not depend on the prompt or its length. Every token at every position is multiplied by the same `W_Q`, `W_K` and `W_V` in a given layer, and each layer has its own set. For "is"(12): its 4,096-number stream vector × `W_Q` gives its 4,096-number Q.

- **All 14 tokens have a Q, a K and a V at each layer.** Even the repeated word "US" (positions 7 and 11) gets different vectors, because it sits at a different position with different words before it.

**What happens when**

Reading the columns:
- **Fed in:** The token whose Q, K and V are computed at this step. Its K and V are added to the cache. Only its Q is used to look things up.
- **Q looks at the K and V of:** Every token up to and including the fed-in one. Keys and values come from the same tokens. The Q is compared with the keys, and the resulting weights blend the values.
- **Output after all layers:** Not a Q, K or V. It is the result of the whole network: the final vector for the fed-in position is turned into scores over the vocabulary, and the top choice is the next token. That token has no Q, K or V yet. It gets them at the next step, when it is fed in.

| Step | Fed in | Q looks at the K and V of (count) | Output after all layers |
|---|---|---|---|
| Prefill | tokens 1-7 together | each token sees itself and earlier ones, so "US" sees 7 | "capital" (taken from the last token, "US") |
| Decode 1 | capital (8) | tokens 1-8 (8) | city |
| Decode 2 | city (9) | tokens 1-9 (9) | of |
| Decode 3 | of (10) | tokens 1-10 (10) | US |
| Decode 4 | US (11) | tokens 1-11 (11) | is |
| Decode 5 | is (12) | tokens 1-12 (12) | washington |
| Decode 6 | washington (13) | tokens 1-13 (13) | DC |

**Zoom in on the step that produces "washington"**

- **Input:** The token "is" (12) is fed in. The model computes its Q, K and V and adds its K and V to the cache.
- **Attention:** "is" uses its Q against the 12 cached keys (tokens 1-12), then takes a weighted average of the matching 12 values.
- **Likely focus:** The heads probably give high weight to "US" and "capital" (positions 4, 7, 8 and 11), since the answer depends on them. This is a guess, not measured. Researchers measure it per head.
- **Prediction:** The blended values are added to the stream of "is". By the last layer, that stream holds the model's guess for the next word, "washington". "washington" itself has no Q, K or V until the next step, when it is fed in.

**What this shows about the roles**

- **Every token plays all three roles.** Its Q asks, later tokens look up its K, and its V gets copied.
- **The only asymmetry is the causal mask.** A token's Q sees only keys at or before its own position.
- **Prompt tokens vs generated tokens:** Both get K and V stored in the cache. The prompt tokens' Q are used once during prefill and then dropped.

**Why online tutorials use the same input and output tokens:** In training, the model sees a whole sequence at once and predicts the next token at every position. The "output" is the same sequence shifted by one position (input `the cat sat`, target `cat sat on`). Tutorials show that training picture. The example above shows generation, where the answer is produced one token at a time.

**Real chat models differ in two ways:** they put special chat-template tokens between the question and the answer, and they would more likely answer "Washington, D.C." than repeat the question.
