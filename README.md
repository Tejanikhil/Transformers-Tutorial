# All about Transformers

* First question that i would like to address is to why transformers rather than RNN architectures ?
-> Traditional RNN models posses vanishing gradient problems which makes it challenging in case of capturing the long range dependencies. 
-> RNN's are sequential in nature which makes it less parallelizable. 
-> Transformers allows for more parallelization than RNN's while training. 
-> Transformers use attention mechanisms which enables them to focus on the relavent parts of the input sequence.

* Low level abstraction of Transformers
-> Transformers are made up of a series of encoder stack and decoder stack
-> Encoder encodes the data into vectors, generally called as embeddings
-> Decoder decodes the encoded sequences into the required sequence

* Higher level abstraction of Transformers
-> Each encoder consists of a MultiHead attention block, a Feed Forward block and a LayerNormalization Block
-> Each decoder consists of a Masked MultiHead attention block, a layer normalization block and a FeedForward Block

* Lets go more deep into it
* What is the significance of attention block ?
-> Self attention mechanism is the one which addresses the problem of long range dependencies. The attention mechanism allows the model to capture dependencies between distant elements in the input sequence

* How attention mechanism are addressing the long range dependencies problem /
-> The attention mechanism in transformers allows the model to consider all elements in the sequence simultaneously, providing a global context that enables the model to capture long-range dependencies more effectively
-> The attention mechanism computes attention scores between all pairs of elements in the input sequence. These attention scores determine how much each element attends to other elements in the sequence
-> More relavent words are assigned more attention weights whereas the less relavent words are assigned the less attention weights.

* How are attention mechanisms different from masked attention mechanisms ?
-> Masked attention mechanisms are primarily used in the decoder layers to enforce causality during training in autoregressive tasks. Specifically, they prevent the model from attending to future elements in the target sequence, ensuring that each token can only attend to previous tokens during generation.

* What are positional embeddings and its significance ?
-> Positional embeddings are additional embeddings added to the input embeddings in transformer architectures to provide information about the position of tokens in the input sequence.

* What are the learnable parameters in transformer models ?
-> Embedding matrices which include token embeddings and positional embeddings
-> Linear Transformation Matrices (used in attention matrices -> query, key and value)
-> FeedForward neural network parameters
-> Layer Normalization parameters which include scaling and shifting params
-> Output Layer parameters
