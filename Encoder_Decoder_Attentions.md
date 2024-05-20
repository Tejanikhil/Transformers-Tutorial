# Encoder Decoder Attention Mechanism

This file dives into the self-attention mechanism.
- Generally, the attention mechanism is used to generate the embeddings by correlating it with the input sequence.

- **Query (Q)**: The word in the output sequence (last token) for which we are calculating the attention score.
- **Key (K)**: Represents the word from the input sequence with which we are attending the query.
- **Value (V)**: The word/vector that is weighted by the attention scores to form the final output.

Here 
Query Q = Masked_Self_Attention(Y) (token on the output sequence) - The Masked self attention embeddings
key K = Ci (input tokens embeddings)

Remaining computation is similar to as mentioned in [self_attention](https://github.com/Tejanikhil/Transformers-Tutorial/edit/main/Self_Attentions.md)
