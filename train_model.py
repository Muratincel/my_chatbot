import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense
import pickle
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import tensorflow as tf


# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Parameters
VOCAB_SIZE = len(tokenizer.word_index) + 1
EMBEDDING_DIM = 256
HIDDEN_UNITS = 512

# Define the encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
encoder_gru = GRU(HIDDEN_UNITS, return_state=True)
encoder_outputs, encoder_state = encoder_gru(encoder_embedding)

# Define the decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
decoder_gru = GRU(HIDDEN_UNITS, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_embedding, initial_state=encoder_state)
decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load tokenized and padded data
data = np.load('tokenized_data.npz')
train_input_seqs = data['train_input_seqs']
train_target_seqs = data['train_output_seqs']
val_input_seqs = data['val_input_seqs']
val_target_seqs = data['val_output_seqs']

print("Loaded tokenized data:")
print("Train input shape:", train_input_seqs.shape)
print("Train target shape:", train_target_seqs.shape)

# Path to save checkpoints
checkpoint_dir = 'checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}.keras")

# Create a checkpoint callback to save the model after each epoch
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, save_weights_only=False, save_best_only=False, save_freq='epoch', verbose=1
)

class StopAfterEpoch(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Automatically stop the training after each epoch
        print(f"Epoch {epoch+1} completed. Saving model and exiting.")
        self.model.stop_training = True

# Check if a model already exists to resume training
if os.path.exists(checkpoint_path.format(epoch=9)):  # check for the last epoch file
    print("Resuming from the last saved epoch...")
    model = tf.keras.models.load_model(checkpoint_path.format(epoch=9))
else:
    print("Starting fresh training...")

# Train the model
model.fit(
    [train_input_seqs, train_target_seqs[:, :-1]],  # Input and decoder input
    train_target_seqs[:, 1:],  # Decoder target
    batch_size=64,
    epochs=10,
    validation_data=(
        [val_input_seqs, val_target_seqs[:, :-1]],
        val_target_seqs[:, 1:]
    ),
    callbacks=[checkpoint_callback, StopAfterEpoch()]
)

# Save the model
model.save('seq2seq_chatbot_gru.keras')

print("Training completed and model saved as 'seq2seq_chatbot_gru.keras'")
