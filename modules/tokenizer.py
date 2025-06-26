from tokenizers import Tokenizer, trainers, models, pre_tokenizers
import os

class BPETokenizer:
    def __init__(self, vocab_size=8000, special_tokens=["<s>", "</s>", "<pad>", "<unk>"]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    #Funzione per allenare il tokenizzatore leggendo i testi da file
    def train(self, files, output_path="tokenizer.json"):
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train(files, trainer)
        self.tokenizer.save(output_path)
        print(f"Tokenizer trained and saved to {output_path}")
    
    #Funzione per ricaricare il tokenizzatore già addestrato
    def load(self, path="tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(path)
        print(f"Tokenizer loaded from {path}")

    #Funzione per codificare il testo in ID
    def encode(self, text):
        return self.tokenizer.encode(text).ids

    #Funzione per decodificare gli ID in testo
    def decode(self, ids):
        return self.tokenizer.decode(ids)

# Test/Debug
if __name__ == "__main__":
    bpe_tokenizer = BPETokenizer()

    #La funzione train viene chiamata solo se il file tokenizer.json non esiste
    if os.path.exists("tokenizer.json"):
        bpe_tokenizer.load()
    else:
        bpe_tokenizer.train(["data/train.en", "data/train.it"])

    test = "Hello world!"
    encoded = bpe_tokenizer.encode(test)
    print("Encoded:", encoded)
    print("Decoded:", bpe_tokenizer.decode(encoded))
