# sample_chat.py
import torch
import os
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Utilisation du device : {device}")

# -----------------------------------------------------------------------------
# Chargement des données pour reconstruire le vocabulaire
# -----------------------------------------------------------------------------
data_dir = "./gemma3"
txt_files = sorted(f for f in os.listdir(data_dir) if f.startswith("data") and f.endswith(".txt"))

if not txt_files:
    raise FileNotFoundError(f"Aucun fichier data*.txt trouvé dans {data_dir}")

full_text = ""
for fname in txt_files:
    with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
        full_text += f.read()

chars = sorted(list(set(full_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi.get(c, 0) for c in s]  # caractères inconnus -> 0
def decode(l: list):
    return ''.join([itos.get(i, "?") for i in l])

# -----------------------------------------------------------------------------
# Chargement du modèle
# -----------------------------------------------------------------------------
ckpt_path = "out/ckpt.pt"
checkpoint = torch.load(ckpt_path, map_location=device)

model = GPT(GPTConfig(**checkpoint["model_args"]))
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

print("✔ Chatbot prêt. Tape 'quit' pour arrêter.")

# -----------------------------------------------------------------------------
# Boucle interactive
# -----------------------------------------------------------------------------
while True:
    question = input("\nVous: ")
    if question.strip().lower() in ["quit", "exit", "q"]:
        break

    prompt = f"Q: {question}\nA:"
    x = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    # Génération
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=200)
    out = decode(y[0].tolist())

    # Extraire uniquement la réponse (après "A:")
    if "A:" in out:
        reponse = out.split("A:", 1)[1].strip()
    else:
        reponse = out[len(prompt):].strip()

    print(f"Bot: {reponse}")
