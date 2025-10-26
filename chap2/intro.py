import re
import tiktoken

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {}

        for k, v in vocab.items():
            self.int_to_str[v] = k

    def encode(self, text):
        preprocessed = re.split(r'([,.!:?;"()_\']|\s|--)', text)
        preprocessed = [c.strip() for c in preprocessed if c.strip()]
        ids = []

        for w in preprocessed:
            if w not in self.str_to_int:
                ids.append(self.str_to_int["<|UNK|>"])
            else:
                ids.append(self.str_to_int[w])

        return ids

    def decode(self, ids):
        text = []

        for id in ids:
            text.append(self.int_to_str[id])

        t = " ".join(text)
        res = re.sub(r'\s+([,.?!"()\'])', r'\1', t)
        return res

# Opening a large tarining file
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total # of chars:", len(raw_text))

# Splitting large text to tokens
preprocessed = re.split(r'([,.!:?;"()_\']|\s|--)', raw_text)
preprocessed = [c.strip() for c in preprocessed if c.strip()]
print("Tokens: ", len(preprocessed))

# Creating token IDs
sorted_words = sorted(set(preprocessed))
sorted_words.extend(["<|UNK|>", "<|endoftext|>"])
print("Vocabulary size:", len(sorted_words))

vocab = {}
for word, i in enumerate(sorted_words):
    vocab[i] = word

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know, Hello" 
           Mrs. Gisburn said with pardonable pride."""

# Try out encode and decode 
ids = tokenizer.encode(text)
print(ids)
orig = tokenizer.decode(ids)
print(orig)

tokenizer = tiktoken.get_encoding("gpt2")
text = "Akwirw ier"
integers = tokenizer.encode(text)
print(integers)
text_back = tokenizer.decode(integers)
print(text_back)
