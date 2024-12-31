# INSTRUCTIONS

## The Scope

First we defined the scope, I choose as a general chatbot which engage in casual conversation

## Collect Data

the dataset is taken from "Cornell Movie Dialogues Dataset"

### Data Verification

we create verify_data.py. here we download the data and checked couple of conversation if the data is correct, and note that it is coming in JSON format. It is structured data

### Input-Response Pairs definition

we create InpRespPairs.py. here we create input_output_pairs dictionary as conversations and this will be used in the next step

### Preprocessing The Text

we create preprocess.py. we are kind of shaping the data by removing unnecessary special characters and extra spaces. we make all the text lowercase as well.

### Splitting The Data

we create split_data.py. we split the data as training and validation pairs. As a result we should get %80 for training and %20 for validating data. this process is to be able to learn, not to memorize the answers

### Tokenization

we create tokenize_chat.py. here we use TensorFlow, our steps: Fitting the tokenizer, converting to sequences,padding sequences and we finally save this tokenizer as tokenizer.pkl file. the reason to do is neural networks work with number, not text.

also tokenized_data.npz is created here (for efficiency purposes, numpy-specific compressed archive format) which stores multiple arrays in a single file for further use

## Model Design

here we implement the Seq2Seq neural network. we chose one of the architectures RNN, GRU, or LSTM (in our case it is GRU)

### Model Training

we create train_model.py we load tokenizer (tokenizer.pkl) define encoder and decoders, compile the model, load the tokenized and padded data (takenized_data.npz) and we train the model by model.fit 


You will only need .npz and .pkl files and train_model.py in order to run. And huge compute power to train..

#### Data size; 
Total Input-Output Pairs: 221616
Training Pairs: 177292, Validation Pairs: 44324

python v10.0

TensorFlow v2.18
