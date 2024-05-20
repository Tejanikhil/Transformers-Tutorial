Description : 
1. Transformers are deep learning models designed to handle sequential data by capturing long-range dependencies. 
2. Introduced by Vaswani et al. in the paper "Attention Is All You Need" (2017),
3. This is the modesl that has revolutionized the field of natural language understanding. 
4. Many large language models including chatGPT uses this architecture as the backbone.
5. This is something that mimics the human understanding.

Lets see how these models learn sequence to sequence tasks

Intuition behind transformers : 
1. Understand the input sequence
2. Correlate the input sequence with the output sequence

How does this model understand the input sequence ? 
* This is acheived using self attention mechanisms.
Lets see how this mechanism helps the model in understanding the input sentence
Input Sequence : a b c d
Output Sequence : x y z

Step1 : These self attention mechanisms first calculates the correlation between the words. 
As 
