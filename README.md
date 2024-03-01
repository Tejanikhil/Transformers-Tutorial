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

Here is how a transformer model looks like

![image](https://github.com/Tejanikhil/Transformers-Tutorial/assets/102232692/63d330ce-0c3c-4358-9469-6aea162db77e)

## Significance of Attention Block
- Attention mechanisms enable capturing dependencies between distant elements in the input sequence.
- The attention mechanism computes attention scores between all pairs of elements in the input sequence, assigning more weight to relevant words.

## How do they acheive it
* Self-Attention: When processing the word "cat," self-attention allows the model to attend to all other words in the sentence, such as "the," "sat," "on," and "mat." The attention weights indicate the relevance of each word to the word "cat" in the context of the entire sentence.
* Multi-Head Attention: Different attention heads may focus on different aspects of the sentence. One attention head may focus on the relationship between "cat" and "mat" (spatial relationship), while another attention head may focus on the relationship between "cat" and "sat" (semantic relationship).
* Mathematically this is acheived using query, keys and values

## What are query, keys and values ? 
* Query:
**Intuition**: The query represents the current token for which we want to compute the attention scores. Think of it as the token that is asking "what should I pay attention to?"
**Example**: In language understanding tasks, if the current token is "cat," the query represents the information or context associated with "cat" that needs to be attended to.
**Mathematically**: The query is a linear transformation of the input embeddings, projecting the input into a lower-dimensional space to facilitate attention computations.
* Key:
**Intuition**: The key provides context or information about other tokens in the sequence. It helps the model understand the relationship between the current token (query) and other tokens in the sequence.
**Example**: In language understanding tasks, if the current token is "cat," the key provides information about other words in the sentence, such as "mat," "sat," "the," etc.
**Mathematically**: Like the query, the key is obtained through a linear transformation of the input embeddings, projecting the input into a lower-dimensional space.
* Value:
**Intuition**: The value represents the actual information or content associated with each token in the sequence. It serves as the "content" that the model will focus on or pay attention to based on the attention scores computed from the query and key.
**Example**: In language understanding tasks, if the current token is "cat," the value provides the actual meaning or content associated with "cat."
**Mathematically**: The value is also obtained through a linear transformation of the input embeddings, projecting the input into a lower-dimensional space.



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
