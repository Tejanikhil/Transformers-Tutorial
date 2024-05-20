Description : 
1. Transformers are deep learning models designed to handle sequential data by capturing long-range dependencies. 
2. It was Introduced by Vaswani et al. in the paper "Attention Is All You Need" (2017) and has revolutionized the field of natural language understanding. Many large language models including chatGPT uses this architecture as their backbone.
5. This is something that mimics the human understanding.

Training Task : 
Lets see how these models learn sequence to sequence tasks

Intuition behind transformers : 
1. Understand the input sequence (Encoder)
2. Correlate the input sequence with the output sequence (Decoder)

What does the word "Understand" refer to in the viewpoint of transformers ?
- Capture the similarity of the words based on the context. (Semantic meaning)
- Knowing the meaning of the word in that context by understading the surrounding words of the phrase. (Contextual meaning)

Lets put this mathematically!!
In mathematics dot product <V1, V2> represents the projection of V1 on V2. In other terms this represents how similar two vectors are.

Lets see how this mechanism helps the model in understanding the input sentence (Encoder)
```
Input Sequence : a b c d
Output Sequence : x y z
```

Step by Step procedure of Encoder:
1. As the model understands only the vector representation, first it converts the sentence into a vector representation called as embeddings as follows
```
Input sequence : a b c d
Embeddings : Ea Eb Ec Ed
```
3. To capture the positional information of the words in the sentence, it adds another vector called as positional embeddings which has positional information.
```Positional Embeddings : Pa Pb Pc Pd```
4. After adding the positional embeddings lets say the embeddings has become Ea', Eb', Ec', Ed'
5. Performs ```self_attention(Ea', Eb', Ec', Ed') -> Ca Cb Cc Cd```
Where Ca, Cb, Cc, Cd represents the contextual embeddings of each word which has the contextual information. 

```Encoder Output = [Ca Cb Cc Cd]```

Before going to the decoder see how the decoder predicts the full sentence 'https://github.com/Tejanikhil/Transformers-Tutorial/edit/main/decoder-overview.md'

* Now lets see how the model correlates the input sequence with the target sequence while learning.
Step by Step Procedure of a Decoder:
1. Decoder initializes with a token [BOS] - begining of the sentence
Given the contextual embeddings of the input sequence ```C = [Ca Cb Cc Cd]```
Lets say the current state of the decoder is [BOS]
* Iteration1 : 
3. Now the decoder has to somehow get the embeddings of the words that it is going to predict next. And that is formulated as below
```y1 = Embedding(y0) + PositionalEmbedding(y1)```
4. Performs Masked Self Attention of y1 to get the contextual embedding of the token (associated with the embedding y1) which is going to be predicted next
```y1' = Layernormalization(y1 + MaskedSelfAttention(y1))```
And now we have the contextual information of y1.
5. Now its time to correlate with the input sequence - Encoder_Decoder_Attention(y1')
In this step, the decoder attends to the entire input sequence (encoded by the encoder) to gather relevant information for generating the current output token.
```y1'' = Layernormalization(y1' + Encoder_Decoder_Attention(y1'))```
6. Project this embedding vector onto the vocabulary and apply softmax function to get the probability distribution over the whole vocabulary
```Y1 = LayerNormalization(y1'' + FeedForwardLayer(y1''))```
7. Probability Distribution : ```P(Y1 | Y0, input_sequence) = softmax(Y1)```

* Iteration2 :
The current state of the decoder is [BOS] x
1. Now the decoder has to somehow get the embeddings of the word that it is going to predict next (token 'y'). And that is formulated as below
```y2 = Embedding(y1) + PositionalEmbedding(y2)```
2. Performs Masked Self Attention of y2 to get the contextual embedding of the token (associated with the embedding y2) which is going to be predicted next
```y2' = Layernormalization(y2 + MaskedSelfAttention(y2))```
And now we have the contextual information of y2.
5. Now its time to correlate with the input sequence - ```Encoder_Decoder_Attention('y2')```
In this step, the decoder attends to the entire input sequence to gather relevant information for generating the current output token (y).
```y2'' = Layernormalization(y2' + Encoder_Decoder_Attention(y2'))```
6. Project this embedding vector onto the vocabulary and apply softmax function to get the probability distribution over the whole vocabulary
```Y2 = LayerNormalization(y2'' + FeedForwardLayer(y2''))```
7. Probability Distribution : ```P(Y2 | Y1, Y0, input_sequence) = softmax(Y2)```

And the iteration continues till the decoder generates the special token [EOS] - end of the sentence
  
