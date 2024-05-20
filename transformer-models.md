# Transformers Overview

## Description

1. Transformers are deep learning models designed to handle sequential data by capturing long-range dependencies.
2. They were introduced by Vaswani et al. in the paper "Attention Is All You Need" (2017) and have revolutionized the field of natural language understanding. Many large language models, including ChatGPT, use this architecture as their backbone.
3. This architecture mimics human understanding.

## Training Task

Let's see how these models learn sequence-to-sequence tasks.

## Intuition Behind Transformers

1. Understand the input sequence (Encoder)
2. Correlate the input sequence with the output sequence (Decoder)

### What Does "Understand" Mean in the Viewpoint of Transformers?

- Capture the similarity of the words based on the context (semantic meaning).
- Know the meaning of the word in that context by understanding the surrounding words of the phrase (contextual meaning).

## Mathematical Explanation

In mathematics, the dot product `<V1, V2>` represents the projection of V1 on V2. In other terms, this represents how similar two vectors are.

Let's see how this mechanism helps the model in understanding the input sentence (Encoder).


```
Input Sequence : a b c d
Output Sequence : x y z
```


### Step-by-Step Procedure of the Encoder

1. As the model understands only the vector representation, it first converts the sentence into a vector representation called embeddings as follows:
    ```
    Input sequence: a b c d
    Embeddings: Ea Eb Ec Ed
    ```
2. To capture the positional information of the words in the sentence, it adds another vector called positional embeddings which has positional information:
    ```
    Positional Embeddings: Pa Pb Pc Pd
    ```
3. After adding the positional embeddings, let's say the embeddings become Ea', Eb', Ec', Ed'.
4. It performs [self attention](https://github.com/Tejanikhil/Transformers-Tutorial/edit/main/Self Attention.md):
    ```
    self_attention(Ea', Eb', Ec', Ed') -> Ca Cb Cc Cd
    ```
    Where Ca, Cb, Cc, Cd represent the contextual embeddings of each word which have the contextual information.

    ```
    Encoder Output = [Ca Cb Cc Cd]
    ```

Before going to the decoder, see how the decoder predicts the full sentence: [Decoder Overview](https://github.com/Tejanikhil/Transformers-Tutorial/edit/main/decoder-overview.md).

### Step-by-Step Procedure of a Decoder

* Now let's see how the model correlates the input sequence with the target sequence while learning.

1. The decoder initializes with a token [BOS] - beginning of the sentence.
   Given the contextual embeddings of the input sequence:
    ```
    C = [Ca Cb Cc Cd]
    ```
   Let's say the current state of the decoder is [BOS].

* **Iteration 1**:
    1. Now the decoder has to somehow get the embeddings of the words that it is going to predict next. This is formulated as below:
        ```
        y1 = Embedding(y0) + PositionalEmbedding(y1)
        ```
    2. It performs masked self-attention of y1 to get the contextual embedding of the token (associated with the embedding y1) which is going to be predicted next:
        ```
        y1' = Layernormalization(y1 + MaskedSelfAttention(y1))
        ```
       And now we have the contextual information of y1.
    3. Now it's time to correlate with the input sequence:
        ```
        Encoder_Decoder_Attention(y1')
        ```
       In this step, the decoder attends to the entire input sequence (encoded by the encoder) to gather relevant information for generating the current output token:
        ```
        y1'' = Layernormalization(y1' + Encoder_Decoder_Attention(y1'))
        ```
    4. Project this embedding vector onto the vocabulary and apply the softmax function to get the probability distribution over the whole vocabulary:
        ```
        Y1 = LayerNormalization(y1'' + FeedForwardLayer(y1''))
        ```
    5. Probability Distribution:
        ```
        P(Y1 | Y0, input_sequence) = softmax(Y1)
        ```

* **Iteration 2**:
    The current state of the decoder is [BOS] x.
    1. Now the decoder has to somehow get the embeddings of the word that it is going to predict next (token 'y'). This is formulated as below:
        ```
        y2 = Embedding(y1) + PositionalEmbedding(y2)
        ```
    2. It performs masked self-attention of y2 to get the contextual embedding of the token (associated with the embedding y2) which is going to be predicted next:
        ```
        y2' = Layernormalization(y2 + MaskedSelfAttention(y2))
        ```
       And now we have the contextual information of y2.
    3. Now it's time to correlate with the input sequence:
        ```
        Encoder_Decoder_Attention(y2')
        ```
       In this step, the decoder attends to the entire input sequence to gather relevant information for generating the current output token (y):
        ```
        y2'' = Layernormalization(y2' + Encoder_Decoder_Attention(y2'))
        ```
    4. Project this embedding vector onto the vocabulary and apply the softmax function to get the probability distribution over the whole vocabulary:
        ```
        Y2 = LayerNormalization(y2'' + FeedForwardLayer(y2''))
        ```
    5. Probability Distribution:
        ```
        P(Y2 | Y1, Y0, input_sequence) = softmax(Y2)
        ```

The iteration continues until the decoder generates the special token [EOS] - end of the sentence.
