# All About Transformers

## Introduction
Transformers have emerged as powerful models for various natural language processing tasks. This README provides an overview of transformers, addressing their advantages over traditional RNN architectures, the structure of transformers, the significance of attention mechanisms, masked attention mechanisms, positional embeddings, and the learnable parameters in transformer models.

## Why Transformers?
- Traditional RNN models suffer from vanishing gradient problems, making it challenging to capture long-range dependencies.
- RNNs are inherently sequential, limiting parallelization during training.
- Transformers allow for more efficient parallelization and utilize attention mechanisms to focus on relevant parts of the input sequence.

## Low-level Abstraction of Transformers
- Transformers consist of encoder and decoder stacks.
- Encoder encodes input data into embeddings.
- Decoder decodes encoded sequences into the required output.

## High-level Abstraction of Transformers
- Each encoder consists of a MultiHead attention block, a Feed Forward block, and a LayerNormalization Block.
- Each decoder consists of a Masked MultiHead attention block, a layer normalization block, and a FeedForward Block.

## Significance of Attention Block
- Attention mechanisms enable capturing dependencies between distant elements in the input sequence.
- The attention mechanism computes attention scores between all pairs of elements in the input sequence, assigning more weight to relevant words.

## Addressing Long-Range Dependencies
- Attention mechanisms allow the model to consider all elements in the sequence simultaneously, capturing long-range dependencies effectively.
- Masked attention mechanisms enforce causality during training, preventing the model from attending to future elements in the target sequence.

## Positional Embeddings and Significance
- Positional embeddings provide information about the position of tokens in the input sequence, addressing the lack of inherent order in transformer architectures.

## Learnable Parameters in Transformer Models
- Embedding matrices (token embeddings and positional embeddings)
- Linear Transformation Matrices (used in attention mechanisms)
- FeedForward neural network parameters
- Layer Normalization parameters (scaling and shifting)
- Output Layer parameters
