import torch
import torch.nn as nn
import math

class Seq2SeqTransformer3(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024,
                 dropout=0.1, activation='relu', layer_norm_eps=1e-5, 
                 max_seq_length=512, tie_weights=False, use_label_smoothing=False):
        super(Seq2SeqTransformer3, self).__init__()
        
        # Parametri del modello
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_length = max_seq_length
        self.tie_weights = tie_weights
        
        # Embedding layers con inizializzazione migliorata
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        
        # Inizializzazione Xavier per gli embedding
        nn.init.xavier_uniform_(self.src_tok_emb.weight)
        nn.init.xavier_uniform_(self.tgt_tok_emb.weight)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer con parametri più flessibili
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True  # Pre-LN (più stabile del post-LN)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True  # Pre-LN
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps)
        )
        
        # Output projection con dropout
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # Weight tying (condivide pesi tra embedding target e output)
        if tie_weights:
            if d_model != tgt_vocab_size:
                raise ValueError(f"Per weight tying, d_model ({d_model}) deve essere uguale a tgt_vocab_size ({tgt_vocab_size})")
            self.fc_out.weight = self.tgt_tok_emb.weight
        
        # Inizializzazione output layer
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0.)
    
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None):
        """
        Args:
            src: Sequenza sorgente [batch_size, src_seq_len]
            tgt: Sequenza target [batch_size, tgt_seq_len]
            src_key_padding_mask: Maschera padding per src [batch_size, src_seq_len]
            tgt_key_padding_mask: Maschera padding per tgt [batch_size, tgt_seq_len]
            memory_key_padding_mask: Maschera padding per memory (di solito uguale a src)
        """
        # Input validation
        if src.size(0) != tgt.size(0):
            raise ValueError(f"Batch size mismatch: src {src.size(0)} vs tgt {tgt.size(0)}")
        
        # Crea maschera causale per il decoder
        tgt_seq_len = tgt.size(1)
        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        # Embedding con scaling
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        
        # Positional encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Encoder
        memory = self.encoder(
            src_emb, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Decoder
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask or src_key_padding_mask
        )
        
        # Output projection con dropout
        output = self.dropout(output)
        output = self.fc_out(output)
        
        return output
    
    def encode(self, src, src_key_padding_mask=None):
        """Encode solo la sequenza sorgente"""
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        return self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
    
    def decode(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, 
               memory_key_padding_mask=None):
        """Decode con memory pre-codificata"""
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = self.decoder(
            tgt_emb, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        output = self.dropout(output)
        return self.fc_out(output)
    
    def _generate_square_subsequent_mask(self, sz):
        """Crea maschera causale ottimizzata"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    
    def create_padding_mask(self, seq, pad_token=0):
        """Crea maschera per padding tokens"""
        return (seq == pad_token)
    
    def get_attention_weights(self, src, tgt, src_key_padding_mask=None, 
                            tgt_key_padding_mask=None, layer_idx=-1):
        """
        Estrae i pesi di attenzione per visualizzazione
        Nota: Richiede modifica del forward per salvare attention weights
        """
        # Implementazione base - per funzionalità complete serve modificare i layer
        with torch.no_grad():
            output = self.forward(src, tgt, src_key_padding_mask, tgt_key_padding_mask)
        return output
    
    def count_parameters(self):
        """Conta parametri trainabili"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        """Calcola dimensione modello in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


class PositionalEncoding(nn.Module):
    """Positional Encoding migliorato con caching"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
        # Pre-calcola tutti i positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Usa una formula più stabile numericamente
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor di input [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequenza troppo lunga: {seq_len} > {self.pe.size(1)}")
        
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing per migliorare la generalizzazione"""
    
    def __init__(self, size, padding_idx=0, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        
    def forward(self, x, target):
        """
        Args:
            x: Predictions [batch_size * seq_len, vocab_size]
            target: Ground truth [batch_size * seq_len]
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(torch.log_softmax(x, dim=1), true_dist)


# Esempio di utilizzo con configurazioni diverse
def create_small_model(vocab_size):
    """Modello piccolo per test rapidi"""
    return Seq2SeqTransformer3(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        dropout=0.4
    )

def create_medium_model(vocab_size):
    """Modello medio per training serio"""
    return Seq2SeqTransformer3(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        tie_weights=True
    )

def create_large_model(vocab_size):
    """Modello grande per risultati migliori"""
    return Seq2SeqTransformer3(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=768,
        nhead=12,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=3072,
        dropout=0.1,
        tie_weights=True
    )