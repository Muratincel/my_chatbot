from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pickle
from split_data import train_pairs, val_pairs
import numpy as np

# Step 1: Fit the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([inp for inp, resp in train_pairs] + [resp for inp, resp in train_pairs])

# Step 2: Convert to sequences
train_input_seqs = tokenizer.texts_to_sequences([inp for inp, resp in train_pairs])
train_output_seqs = tokenizer.texts_to_sequences([resp for inp, resp in train_pairs])

val_input_seqs = tokenizer.texts_to_sequences([inp for inp, resp in val_pairs])
val_output_seqs = tokenizer.texts_to_sequences([resp for inp, resp in val_pairs])

# Step 3: Pad sequences
max_len = max(max(len(seq) for seq in train_input_seqs), max(len(seq) for seq in train_output_seqs))
train_input_seqs = pad_sequences(train_input_seqs, maxlen=max_len, padding='post')
train_output_seqs = pad_sequences(train_output_seqs, maxlen=max_len, padding='post')
val_input_seqs = pad_sequences(val_input_seqs, maxlen=max_len, padding='post')
val_output_seqs = pad_sequences(val_output_seqs, maxlen=max_len, padding='post')

# Step 4: Save the tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Tokenization and padding completed! Max sequence length: {max_len}")

np.savez_compressed(
    'tokenized_data.npz',
    train_input_seqs=train_input_seqs,
    train_output_seqs=train_output_seqs,
    val_input_seqs=val_input_seqs,
    val_output_seqs=val_output_seqs
)

print("Tokenized and padded sequences saved to 'tokenized_data.npz'")

#! Validating the matrix issue

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from InpRespPairs import input_output_pairs

# def tokenize_and_vectorize(input_output_pairs):
#     inputs = [pair[0] for pair in input_output_pairs]  # Input sentences
#     outputs = [pair[1] for pair in input_output_pairs]  # Output sentences
    
#     # Use TF-IDF Vectorizer to convert text into numerical data
#     vectorizer = TfidfVectorizer(max_features=10000)
#     X = vectorizer.fit_transform(inputs)  # Feature matrix (inputs)
#     Y = vectorizer.transform(outputs)     # Target matrix (outputs)
    
#     return X, Y, vectorizer

# X, Y, vectorizer = tokenize_and_vectorize(input_output_pairs)

# import numpy as np
# from scipy.sparse import issparse

# def validate_matrix(X, Y):
#     print("Validating matrices...")

#     def check_sparse_matrix(matrix, name):
#         if issparse(matrix):
#             data = matrix.data
#         else:
#             data = matrix
#         if np.any(np.isnan(data)):
#             print(f"{name} contains NaN values.")
#         elif np.any(np.isinf(data)):
#             print(f"{name} contains Infinity values.")
#         else:
#             print(f"{name} is valid (no NaN or Infinity).")

#     check_sparse_matrix(X, "X (input matrix)")
#     check_sparse_matrix(Y, "Y (output matrix)")

#     XT_X = (X.T @ X).tocsc()
#     rank = np.linalg.matrix_rank(XT_X.toarray())
#     print(f"Rank of X^T X: {rank} (Expected: {XT_X.shape[0]})")
#     if rank < XT_X.shape[0]:
#         print("Warning: Feature matrix X may have linearly dependent columns.")


# validate_matrix(X, Y)

# # Print shapes of matrices
# print(f"Shape of X (input feature matrix): {X.shape}")
# print(f"Shape of Y (output target matrix): {Y.shape}")

# # Inspect features and sample rows
# print("Sample features for X (input):")
# print(vectorizer.get_feature_names_out()[:10])  # First 10 feature names
# print("Sample row from X (input feature matrix):", X[0])
# print("Sample row from Y (output target matrix):", Y[0])