import random
from sklearn.model_selection import train_test_split

def create_sample_subset(src_file, tgt_file, src_mini_file, tgt_mini_file, ratio=0.05):
    # Leggi il dataset completo
    with open(src_file, encoding="utf-8") as f:
        src_lines = f.readlines()

    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = f.readlines()

    # Assicura che siano allineati
    assert len(src_lines) == len(tgt_lines), "Mismatch nel numero di righe"

    # Estrai casualmente il 5% dei dati
    src_sample, _, tgt_sample, _ = train_test_split(src_lines, tgt_lines, train_size=ratio, random_state=42)

    # Salva i campioni ridotti
    with open(src_mini_file, "w", encoding="utf-8") as f:
        f.writelines(src_sample)

    with open(tgt_mini_file, "w", encoding="utf-8") as f:
        f.writelines(tgt_sample)

    print(f"Campioni salvati: {len(src_sample)} righe")

def create_val_split(src_file, tgt_file, src_splitted_file, tgt_splitted_file, src_val_file, tgt_val_file, val_ratio=0.2):

    # Leggi le frasi originali e tradotte
    with open(src_file, encoding="utf-8") as f:
        src_lines = f.readlines()

    with open(tgt_file, encoding="utf-8") as f:
        tgt_lines = f.readlines()

    # Assicura che siano allineate
    assert len(src_lines) == len(tgt_lines), "Numero di righe non corrisponde"

    # Suddividi 80% train, 20% val
    src_train, src_val, tgt_train, tgt_val = train_test_split(src_lines, tgt_lines, test_size=val_ratio, random_state=42)

    # Scrivi su file
    with open(src_splitted_file, "w", encoding="utf-8") as f:
        f.writelines(src_train)
    with open(tgt_splitted_file, "w", encoding="utf-8") as f:
        f.writelines(tgt_train)

    with open(src_val_file, "w", encoding="utf-8") as f:
        f.writelines(src_val)
    with open(tgt_val_file, "w", encoding="utf-8") as f:
        f.writelines(tgt_val)

    print("Divisione completata.")

'''
def create_train_val_split(en_file, it_file, val_ratio=0.1, random_seed=42):
    """
    Crea split train/validation da file paralleli
    
    Args:
        en_file: percorso file inglese
        it_file: percorso file italiano
        val_ratio: percentuale per validazione (0.1 = 10%)
        random_seed: seed per riproducibilità
    """
    
    # Leggi i file
    print("Lettura file...")
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()
    
    with open(it_file, 'r', encoding='utf-8') as f:
        it_lines = f.readlines()
    
    # Verifica che abbiano lo stesso numero di righe
    assert len(en_lines) == len(it_lines), f"Numero righe diverso: EN={len(en_lines)}, IT={len(it_lines)}"
    
    total_lines = len(en_lines)
    val_size = int(total_lines * val_ratio)
    train_size = total_lines - val_size
    
    print(f"Righe totali: {total_lines}")
    print(f"Training: {train_size} righe ({(1-val_ratio)*100:.1f}%)")
    print(f"Validation: {val_size} righe ({val_ratio*100:.1f}%)")
    
    # Crea indici e mescola
    random.seed(random_seed)
    indices = list(range(total_lines))
    random.shuffle(indices)
    
    # Split indici
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Crea file di training
    print("Creazione file di training...")
    with open('data/train.en', 'w', encoding='utf-8') as f:
        for i in train_indices:
            f.write(en_lines[i])
    
    with open('data/train.it', 'w', encoding='utf-8') as f:
        for i in train_indices:
            f.write(it_lines[i])
    
    # Crea file di validazione
    print("Creazione file di validazione...")
    with open('data/val.en', 'w', encoding='utf-8') as f:
        for i in val_indices:
            f.write(en_lines[i])
    
    with open('data/val.it', 'w', encoding='utf-8') as f:
        for i in val_indices:
            f.write(it_lines[i])
    
    print("✅ File creati con successo!")
    print("📁 File generati:")
    print("   - data/train.en")
    print("   - data/train.it") 
    print("   - data/val.en")
    print("   - data/val.it")
'''

# Utilizzo
if __name__ == "__main__":
    # I tuoi file originali (fai backup prima!)
    original_en = "data/train_original.en"
    original_it = "data/train_original.it"
    
    # File per il campione ridotto
    mini_en = "data/train_mini.en"
    mini_it = "data/train_mini.it"

    # File per la suddivisione train/val
    train_en = "data/train.en"
    train_it = "data/train.it"
    val_en = "data/val.en"
    val_it = "data/val.it"

    # Crea il campione ridotto
    create_sample_subset(original_en, original_it, mini_en, mini_it)

    # Crea la suddivisione train/val
    create_val_split(mini_en, mini_it, train_en, train_it, val_en, val_it)
    print("Operazioni completate con successo!")