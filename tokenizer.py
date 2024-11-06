import json

PUNCTUATION = ["!", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`"]

SPACE = (" ", "Ġ")

END_OF_FILE = "<|endoftext|>"

class Tokenizer():
    def __init__(self, vocab_file="vocab-5000.json"):
        with open(vocab_file) as f:
            self.vocab = json.load(f)
        self.end_of_file_index = self.vocab.index(END_OF_FILE)
        
    def tokenize(self, text, add_eof=True):
        tokens = []
        for word in text.split():
            if word not in PUNCTUATION:
                word = SPACE[1] + word
            i = 0
            while i < len(word):
                j = i + 1
                index = self.vocab.index(word[i:j])
                while (j <= len(word)):
                    if word[i:j] in self.vocab:
                        index = self.vocab.index(word[i:j])
                    else:
                        break
                    j +=1
                tokens.append(index)
                i = j-1

        if add_eof:
            tokens.append(self.end_of_file_index)
        return tokens

    def decode(self, tokens):
        string = ""
        for token in tokens:
            word = self.vocab[token]
            string += word.replace(SPACE[1], SPACE[0])
        
        return string
