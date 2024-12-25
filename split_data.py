from sklearn.model_selection import train_test_split
from preprocess import preprocessed_pairs

train_pairs, val_pairs = train_test_split(preprocessed_pairs, test_size=0.2, random_state=42)
print(f"Training Pairs: {len(train_pairs)}, Validation Pairs: {len(val_pairs)}")


# ! FOR SCALING, AGAIN. BELOW CODE (generate seperately train_pairs and val_pairs);

# import pickle

# with open("train_pairs.pkl", "wb") as f:
#     pickle.dump(train_pairs, f)
# with open("val_pairs.pkl", "wb") as f:
#     pickle.dump(val_pairs, f)
