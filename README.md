# Tranformer model using PyTorch
### Overview
This project presents a decoder-only transformer architecture trained on Shakespeare dataset. The goal was to develop a model capable of generating coherent and contextually relevant text in the style of William Shakespeare. The transformer, implemented from scratch, showcases my expertise in natural language processing and transformer architectures.

### Model and Training

This is a decoder only architecture similar to GPT2 model. Model has attension layers, multiple attension heads, whcih can be configured from the config file in the `trainer.py`. In the current version of the model 6 layers and 6 attention heads are used. Embeddign dimension used was 384, context length and the batch size are 256, 64 respectively. Learning rate used was 3e-4. Model trained for 7000 steps and acheived 1.4 validation loss. Sample generation can be seen below. In this case model is trained on Shakespeare text using charecter level encoding. vocab size of 65 allowed me to train with batch size 64 on my single GPU of 6 GB machine. Model has the size of 10 Million parameters. Details model size by layer give below.
#### Model size

Embedding dimension and the vocab size decides the number of parameters in the embedding layer. Also, batch size * context_length * vocab_size in the final classification head decides the memory needs to store the lgits. in total for the above mentioned configuration, total model size in parameters in 10 Million. Breakdown of each layer and block is given below.
<br><br>
![model_size](images/model_size.png)
<br><br>

### Results
generated text sample from the model for the prompt "O God O God!"

![sample_generation](images/smaple_generation.png)

