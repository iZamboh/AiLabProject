import torch
from torch.utils.data import DataLoader
from modules.dataset import load_dataset
from modules.tokenizer import BPETokenizer
from modules.utils import collate_fn
from modules.model import Seq2SeqTransformer
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import modules.utils as utils
import modules.model3 as model3

# Configurazione
SRC_PATH = "data/train.en" # Percorso del file di testo in lingua di origine
TGT_PATH = "data/train.it" # Percorso del file di testo in lingua di destinazione
VAL_SRC_PATH = "data/val.en" # Percorso del file di testo di validazione in lingua di origine
VAL_TGT_PATH = "data/val.it" # Percorso del file di testo di validazione in lingua di destinazione
TOKENIZER_PATH = "tokenizer.json" # Percorso del file del tokenizer
MAX_LENGTH = 128 # Lunghezza massima delle sequenze
BATCH_SIZE = 32 # Dimensione del batch
EPOCHS = 15 # Numero di epoche per il training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Imposta il dispositivo per il training (GPU se disponibile, altrimenti CPU)
CHECKPOINT_DIR = "checkpoints"  # Directory per salvare i checkpoint
SAVE_EVERY = 2  # Salva checkpoint ogni N epoche
PATIENCE = 3  # Numero di epoche senza miglioramento prima di fermare l'allenamento
DATA_LIMIT = 0.05  # Percentuale di dati da utilizzare (1.0 = 100%, 0.5 = 50%, etc.)

# Crea directory per i checkpoint
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Verifica informazioni GPU
print(f"Utilizzo dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU disponibile: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Carica tokenizer 
tokenizer = BPETokenizer()
tokenizer.load(TOKENIZER_PATH)

# Dataset e DataLoader
print("Caricamento dataset...")
train_dataset = load_dataset(SRC_PATH, TGT_PATH, tokenizer, max_length=MAX_LENGTH)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Dataset di validazione
try:
    val_dataset = load_dataset(VAL_SRC_PATH, VAL_TGT_PATH, tokenizer, max_length=MAX_LENGTH)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Dataset di validazione caricato: {len(val_dataset)} esempi")
except FileNotFoundError:
    print("File di validazione non trovati. Continuando solo con il training set.")
    val_dataloader = None

print(f"Dataset di training caricato: {len(train_dataset)} esempi")
print(f"Numero di batch per epoca: {len(train_dataloader)}")

# Modello
vocab_size = tokenizer.vocab_size
#model = Seq2SeqTransformer(vocab_size, vocab_size).to(DEVICE)
model = model3.create_small_model(vocab_size).to(DEVICE)  # Usa il modello più piccolo per il training

# Ottimizzatore, loss e scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience = PATIENCE, factor=0.5)

# Training loop
print("\nInizio training...")
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    # Barra di progresso per l'epoca
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_idx, (src, tgt) in enumerate(pbar):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        logits = model(src, tgt_input)
        logits = logits.reshape(-1, logits.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping per stabilità
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        epoch_loss += loss.item()
        
        # Aggiorna barra di progresso
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Libera memoria GPU periodicamente
        if DEVICE.type == 'cuda' and batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Calcola loss media dell'epoca
    avg_train_loss = epoch_loss / len(train_dataloader)
    
    # Validazione
    if val_dataloader is not None:
        avg_val_loss = utils.validate_model(model, val_dataloader, criterion, DEVICE)
        print(f"\nEpoch {epoch+1} completata:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        
        # Aggiorna learning rate
        scheduler.step(avg_val_loss)
        
        # Salva migliore modello
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            utils.save_checkpoint(
                epoch, model, optimizer, avg_val_loss, scheduler, vocab_size,
                os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            )
        
        current_loss = avg_val_loss
    else:
        print(f"\nEpoch {epoch+1} completata. Training Loss: {avg_train_loss:.4f}")
        scheduler.step(avg_train_loss)
        current_loss = avg_train_loss
    
    # Salva checkpoint 
    utils.save_checkpoint(
        epoch, model, optimizer, current_loss, scheduler, vocab_size,
        os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
    )
    
    # Stampa learning rate corrente
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Learning Rate: {current_lr:.6f}")
    
    # Libera memoria GPU alla fine dell'epoca
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

# Salva modello finale
utils.save_checkpoint(
    EPOCHS-1, model, optimizer, current_loss, scheduler, vocab_size,
    os.path.join(CHECKPOINT_DIR, 'final_model.pth')
)

print(f"\nTraining completato!")
print(f"Modelli salvati in: {CHECKPOINT_DIR}")
if val_dataloader is not None:
    print(f"Migliore validation loss: {best_val_loss:.4f}")