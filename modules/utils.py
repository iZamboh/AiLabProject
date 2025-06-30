import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Prende frasi con lunghezze diverse e le padda (riempie con zeri) per creare batch uniformi.
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded

#Conta il numero di parametri allenabili nel modello.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Funzione per salvare checkpoint
def save_checkpoint(epoch, model, optimizer, loss, scheduler, vocab_size, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'vocab_size': vocab_size,
    }, filepath)
    print(f"Checkpoint salvato: {filepath}")

# Funzione per validazione
def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc="Validazione", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            logits = logits.reshape(-1, logits.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            
            loss = criterion(logits, tgt_output)
            val_loss += loss.item()
            total_samples += src.size(0)
    
    return val_loss / len(val_dataloader)
