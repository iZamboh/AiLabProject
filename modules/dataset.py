import torch
import random
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, tokenizer, max_length=256):
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text = self.src_lines[idx].strip()
        tgt_text = self.tgt_lines[idx].strip()

        src_ids = self.tokenizer.encode(src_text)[:self.max_length]
        tgt_ids = self.tokenizer.encode(tgt_text)[:self.max_length]

        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

        return src_tensor, tgt_tensor

'''
def load_dataset(src_path, tgt_path, tokenizer, max_length=256, data_limit=1.0, seed=42):
    #Legge i file con le frasi della lingua di origine e della lingua di destinazione
    with open(src_path, encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = f.readlines()

    #Controlla che i due file abbiano lo stesso numero di righe
    assert len(src_lines) == len(tgt_lines), "I due file devono avere lo stesso numero di righe."

    # Limita il numero di frasi se richiesto
    if data_limit < 1.0:
        total_lines = len(src_lines)
        limit = int(total_lines * data_limit)
        random.seed(seed)
        indices = random.sample(range(len(src_lines)), limit)
        src_lines = [src_lines[i] for i in indices]
        tgt_lines = [tgt_lines[i] for i in indices]

    return TranslationDataset(src_lines, tgt_lines, tokenizer, max_length)
'''

def load_dataset(src_path, tgt_path, tokenizer, max_length=256):
    #Legge i file con le frasi della lingua di origine e della lingua di destinazione
    with open(src_path, encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(tgt_path, encoding="utf-8") as f:
        tgt_lines = f.readlines()

    #Controlla che i due file abbiano lo stesso numero di righe
    assert len(src_lines) == len(tgt_lines), "I due file devono avere lo stesso numero di righe."

    return TranslationDataset(src_lines, tgt_lines, tokenizer, max_length)