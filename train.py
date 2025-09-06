# train.py
# Boucle d'entraînement de nanoGPT adaptée pour fichiers texte multiples (gemma3/data*.txt)
# Optimisée pour réduire la consommation mémoire
# Licence MIT

import os
import time
import math
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from contextlib import nullcontext

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Hyperparamètres
# -----------------------------------------------------------------------------
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch', 'resume'

# Modèle
block_size = 128
batch_size = 64
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# Optimisation
learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Entraînement sur : {device}")

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float16
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Chargement des données depuis gemma3/data*.txt
# -----------------------------------------------------------------------------
data_dir = "./gemma3"
txt_files = sorted(f for f in os.listdir(data_dir) if f.startswith("data") and f.endswith(".txt"))

if not txt_files:
    raise FileNotFoundError(f"Aucun fichier data*.txt trouvé dans {data_dir}")

print(f"✔ Fichiers trouvés : {txt_files}")

# Lecture de tous les fichiers et concaténation
full_text = ""
for fname in txt_files:
    path = os.path.join(data_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        full_text += f.read()

print(f"✔ Longueur totale du texte : {len(full_text):,} caractères")

# Vocabulaire
chars = sorted(list(set(full_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encodage
data = torch.tensor(encode(full_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -----------------------------------------------------------------------------
# Batch loader optimisé (direct sur device)
# -----------------------------------------------------------------------------
def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,), device=device)
    x = torch.stack([data_[i:i+block_size] for i in ix]).to(device, non_blocking=True)
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix]).to(device, non_blocking=True)
    return x, y

# -----------------------------------------------------------------------------
# Évaluation de la perte optimisée (faible mémoire)
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        total_loss = 0.0
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            total_loss += loss.item()
            del X, Y, loss
            if device == "cuda":
                torch.cuda.empty_cache()
        out[split] = total_loss / eval_iters
    model.train()
    return out

# -----------------------------------------------------------------------------
# Création ou chargement du modèle
# -----------------------------------------------------------------------------
if init_from == 'scratch':
    print("Initialisation du modèle à partir de zéro")
    gptconf = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Reprise depuis {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(checkpoint['model_args'])
    model.load_state_dict(checkpoint['model'])
else:
    raise ValueError(f"init_from inconnu : {init_from}")

model.to(device)

# -----------------------------------------------------------------------------
# Optimiseur
# -----------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

# -----------------------------------------------------------------------------
# Boucle d'entraînement
# -----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

while True:
    if iter_num > max_iters:
        break

    if iter_num % eval_interval == 0 or iter_num == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'model_args': model.config.__dict__,
                'iter_num': iter_num,
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            os.makedirs(out_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    X, Y = get_batch('train')

    with ctx:
        logits, loss = model(X, Y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if iter_num % log_interval == 0:
        print(f"iter {iter_num}: train loss {loss.item():.4f}")

    iter_num += 1
