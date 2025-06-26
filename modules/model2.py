import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, maxlen=5000):
        super().__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

class TransformerMT(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_tgt, emb_size=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.src_tok_emb = nn.Embedding(vocab_size_src, emb_size)
        self.tgt_tok_emb = nn.Embedding(vocab_size_tgt, emb_size)
        self.pos_encoder = PositionalEncoding(emb_size, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.generator = nn.Linear(emb_size, vocab_size_tgt)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.pos_encoder(self.src_tok_emb(src)).transpose(0, 1)
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt)).transpose(0, 1)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=None)
        outs = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=None)
        return self.generator(outs.transpose(0, 1))
