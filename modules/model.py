import torch
import torch.nn as nn
import math

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, num_layers=3, dropout=0.1):  # Cambbiato d_model da 256 a 512, nhead da 4 a 8, num_layers da 2 a 4
        super(Seq2SeqTransformer, self).__init__()

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # Crea maschera causale solo per il target (decoder)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Embedding con scaling (come nel paper originale)
        src = self.src_tok_emb(src) * math.sqrt(self.d_model)
        tgt = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)

        # Aggiungi positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        # ENCODER-DECODER: src va all'encoder, tgt al decoder
        output = self.transformer(
            src=src,                                   # Input encoder
            tgt=tgt,                                   # Input decoder
            tgt_mask=tgt_mask,                         # Maschera causale per decoder
            src_key_padding_mask=src_key_padding_mask, # Maschera padding encoder
            tgt_key_padding_mask=tgt_key_padding_mask  # Maschera padding decoder
        )
        
        return self.fc_out(output)

    # Crea maschera causale per il decoder (per evitare che il modello guardi avanti)
    def _generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    
    # Crea maschera per i token di padding
    def create_padding_mask(self, seq, pad_token=0):
        return (seq == pad_token)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)