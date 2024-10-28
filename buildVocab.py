from data.bookCorpus import Bookcorpus
from collections import defaultdict
import json

builder = Bookcorpus()

builder.download_and_prepare()

ds = builder.as_dataset(split='train[:10%]')

PUNCTUATION = ["!", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`"]

SPACE = (" ", "Ġ")

END_OF_FILE = "<|endoftext|>"

def getWordFreqs(save_file="wordFreqs.json"):
    word_freqs = defaultdict(int)
    for row in ds["text"]:
        for word in row.split():
            if word not in PUNCTUATION:
                word = "Ġ" + word
            if word in word_freqs:
                word_freqs[word] += 1
            else:
                word_freqs[word] = 1
    with open(save_file, "w") as outfile: 
        json.dump(word_freqs, outfile)
    return word_freqs

    #word_freqs = None

#word_freqs = getWordFreqs()

def createTokens(vocab_size=5000, vocab_file_name="vocab-5000.json", word_freqs=None):
    if word_freqs == None:
        with open("wordFreqs.json") as f:
            word_freqs = json.load(f)

    alphabet = []

    for word in word_freqs.keys():
        for c in word:
            if c not in alphabet:
                alphabet.append(c)

    alphabet.sort()

    vocab = alphabet.copy() + [END_OF_FILE]

    splits = {word: [c for c in word] for word in word_freqs.keys()}

    while len(vocab) < vocab_size:
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) > 1:
                for i in range(len(split) - 1):
                    pair = (split[i], split[i+1])
                    if pair in pair_freqs:
                        pair_freqs[pair] += freq
                    else:
                        pair_freqs[pair] = freq


        #print(len(pair_freqs.keys()))
        best_pair = None
        highest_freq = 0
        for pair, freq in pair_freqs.items():
            if freq > highest_freq:
                best_pair = pair
                highest_freq = freq

        if highest_freq == 0:
            break

        print(best_pair[0] + best_pair[1])
        vocab.append(best_pair[0] + best_pair[1])

        for word, split in splits.items():
            if len(split) > 1:
                new_split = []
                for i in range(len(split)):
                    if i != len(split) - 1 and split[i] == best_pair[0] and split[i+1] == best_pair[1]:
                        new_split.append(best_pair[0] + best_pair[1])
                    elif i == 0 or split[i - 1] != best_pair[0] or split[i] != best_pair[1]:
                        new_split.append(split[i])
                splits[word] = new_split


    with open(vocab_file_name, "w") as outfile: 
        json.dump(vocab, outfile)


createTokens()


            


        
        

    
