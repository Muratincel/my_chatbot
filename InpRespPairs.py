from convokit import Corpus, download

corpus = Corpus(filename="C:/Users/Murat/.convokit/saved-corpora/movie-corpus")

input_output_pairs = []

for convo in corpus.iter_conversations():
    utterances = list(convo.iter_utterances())
    for i in range(len(utterances) - 1):  # Pair each utterance with the next
        input_text = utterances[i].text
        response_text = utterances[i + 1].text
        input_output_pairs.append((input_text, response_text))

print(f"Total Input-Output Pairs: {len(input_output_pairs)}")
# print(f"Sample Pair: {input_output_pairs[0]}")


#! WHEN YOU FEED YOUR MODEL WITH OTHER TYPES OF DATA FOR SCALING, USE BELOW APPROACH;

# import pickle

# with open("input_output_pairs.pkl", "wb") as f:
#     pickle.dump(input_output_pairs, f)
