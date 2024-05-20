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
Input Sequence : a b c d
Output Sequence : x y z

Step by Step procedure of Encoder:
1. As the model understands only the vector representation, first it converts the sentence into a vector representation called as embeddings as follows
Input sequence : a b c d
Embeddings : Ea Eb Ec Ed
2. To capture the positional information of the words in the sentence, it adds another vector called as positional embeddings which has positional information.
Positional Embeddings : Pa Pb Pc Pd
3. After adding the positional embeddings lets say the embeddings has become Ea', Eb', Ec', Ed'
4. Performs self_attention(Ea', Eb', Ec', Ed') -> Ca Cb Cc Cd
Where Ca, Cb, Cc, Cd represents the contextual embeddings of each word which has the contextual information. 

Encoder Output = [Ca Cb Cc Cd]

Before going to the decoder see how the decoder predicts the full sentence

* Now lets see how the model correlates the input sequence with the target sequence while learning.
Step by Step Procedure of a Decoder:
1. Decoder initializes with a token [BOS] - begining of the sentence
Now we have the contextual embeddings C = [Ca Cb Cc Cd]
So the current state of the decoder is y0 (i.e) [BOS]
2. Now the decoder has to somehow get the embeddings of the words that it is going to predict next. And that is formulated as below
yt = Embedding(y_t-1) + PositionalEmbeddings(yt)
for the 1st word Embedding(y0) + PositionalEmbedding(1)
for 2nd word Embedding(y1) + PositionalEmbedding(2)
3. Masked Self Attention for yt

Step1 : These self attention mechanisms first calculates the correlation between the words in the input sequence

