# Translation Model using Transformer from scratch

# Description:

## Machine Translation
Machine Translation (MT) is the task of automatically converting one natural language into another, preserving the meaning of the input text, and producing fluent text in the output language.


# Project Structure

1.For the purpose of this project,I have used online dataset of english and hindi sentences.

## Model Aerchitecture 

### Attention Mechanism
1.    In psychology, attention is the cognitive process of selectively concentrating on one or a few things while ignoring others.
      A neural network is considered to be an effort to mimic human brain actions in a simplified manner. Attention Mechanism is also 
      an attempt to implement the same action of selectively concentrating on a few relevant things, while ignoring others in deep neural networks. 
2.    The attention mechanism emerged as an improvement over the encoder decoder-based neural machine translation system in natural language processing 
      (NLP). Later, this mechanism, or its variants, was used in other applications, including computer vision, speech processing, etc.
      
### Understanding the Attention Mechanism
<p align="center">
  <img src="https://user-images.githubusercontent.com/55678844/150070973-79d5fd02-4f2b-4b88-bf07-3dddd360deac.jpg" />
</p>

This is the diagram of the Attention model shown in Bahdanau’s paper. The Bidirectional LSTM used here generates a sequence of annotations (h1, h2,….., hTx) for each input sentence. All the vectors h1,h2.., etc., used in their work are basically the concatenation of forward and backward hidden states in the encoder.

<p align="center">
  <img src="https://user-images.githubusercontent.com/55678844/150071258-bc0298e5-3ff0-4464-bce4-050894e40be2.jpg" />
</p>


To put it in simple terms, all the vectors h1,h2,h3…., hTx are representations of Tx number of words in the input sentence. In the simple encoder and decoder model, only the last state of the encoder LSTM was used (hTx in this case) as the context vector.

But Bahdanau et al put emphasis on embeddings of all the words in the input (represented by hidden states) while creating the context vector. They did this by simply taking a weighted sum of the hidden states.

The context vector ci for the output word yi is generated using the weighted sum of the annotations:

<p align="center">
  <img src="https://user-images.githubusercontent.com/55678844/150071393-2ee9660e-991d-4d88-93a9-d2564a164e21.jpg" />
</p>

 The weights αij are computed by a softmax function given by the following equation:
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/55678844/150071553-49aabd76-7a99-4720-a509-f85ecb4e32e9.jpg" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/55678844/150071656-f63d7e2f-5003-4185-9599-02c0ca1d0d8d.jpg" />
</p>

eij is the output score of a feedforward neural network described by the function a that attempts to capture the alignment between input at j and output at i.

Basically, if the encoder produces Tx number of “annotations” (the hidden state vectors) each having dimension d, then the input dimension of the feedforward network is 
(Tx , 2d) (assuming the previous state of the decoder also has d dimensions and these two vectors are concatenated). This input is multiplied with a matrix Wa of (2d, 1) dimensions (of course followed by addition of the bias term) to get scores eij (having a dimension (Tx , 1)).

On the top of these eij scores, a tan hyperbolic function is applied followed by a softmax to get the normalized alignment scores for output j:

      E = I [Tx*2d] * Wa [2d * 1] + B[Tx*1]

      α = softmax(tanh(E))

      C= IT * α!


So, α is a (Tx, 1) dimensional vector and its elements are the weights corresponding to each word in the input sentence.


### Sequence to Sequence Modelling
![seq_2_seq](https://user-images.githubusercontent.com/55678844/149960315-3e1f8269-0303-44c4-aa8e-5a54ee75c8d3.png)

1.  Sequence Modelling problems refer to the problems where either the input and/or the output is a sequence of data (words, letters…etc.)
2.  Sequence-to-Sequence (Seq2Seq) problems is a special class of Sequence Modelling Problems in which both, the input and the output is a sequence. 
Encoder-Decoder models were originally built to solve such Seq2Seq problems. 

### Encoder-Decoder Model
![encoder_decoder_model](https://user-images.githubusercontent.com/55678844/149959954-099b3ef4-3690-4ae9-98c9-d931db4e4cc8.png)

1. An Encoder-Decoder architecture was developed where an input sequence was read in entirety and encoded to a fixed-length internal representation.
   A decoder network then used this internal representation to output words until the end of sequence token was reached. 
2. Recurrent networks are used for both the encoder and decoder
3. I have used LSTM for this task.

### The Encoder Block
   The encoder part is an GRU cell. It is fed in the input-sequence over time and it tries to encapsulate all its information and store it in its 
   final internal states hₜ (hidden state) . The internal states are then passed onto the decoder part, which it will use to try 
   to produce the target-sequence. This is the ‘context vector’ which we were earlier referring to.The outputs at each time-step of the encoder part 
   are all discarded.
      
### The Decoder Block
   The decoder block is also an GRU cell. The main thing to note here is that the initial states (h₀) of the decoder are set to the final states 
   (hₜ) of the encoder.These act as the ‘context’ vector and help the decoder produce the desired target-sequence.Now the way decoder works, is, that 
   its output at any time-step t is supposed to be the tᵗʰ word in the target-sequence/Y_true. 
## Future Improvements
   Developed this model using character level classification approach due to hardware/memory constraint.In future we can use word level classification which makes the prediction more accurate.
   
