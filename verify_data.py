from convokit import Corpus, download


# for the first time you need to download and load the Cornell Movie Dialogues dataset with below code
# corpus = Corpus(filename=download('movie-corpus'))

# ~ in my case i just read from my local
corpus = Corpus(filename="C:/Users/Murat/.convokit/saved-corpora/movie-corpus")

for i, convo in enumerate(corpus.iter_conversations()):
    print(f"Conversation {i+1}:")
    for utterance in convo.iter_utterances():
        print(f"  {utterance.speaker.id}: {utterance.text}")
    if i == 5:  # Print only the first 3 conversations for brevity
        break

"""now you are in the 1st step, run this code and lets see, the other
steps are explained in the chat""" 

# 1. Verify Dataset Contents
# 2. Define Input-Response Pairs
# 3. Preprocess Text
# 4. Split Data
# 5. Next Step: Model Design
