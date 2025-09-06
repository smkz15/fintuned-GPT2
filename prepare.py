# prepare_stream.py
# Télécharge "stingning/ultrachat" en plusieurs fichiers (2Go max chacun)
# Construit vocabulaire et encode en streaming (faible RAM)
# Usage: python3 prepare_stream.py

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

HERE = os.path.dirname(__file__)
os.makedirs(HERE, exist_ok=True)

META_PKL = os.path.join(HERE, "meta.pkl")
TRAIN_PKL = os.path.join(HERE, "train.pkl")
VAL_PKL = os.path.join(HERE, "val.pkl")

MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 Go par fichier
TARGET_LINES = None  # None = toutes

def get_file_path(index):
    return os.path.join(HERE, f"data{index}.txt")

def extract_full_conversation(ex):
    if "data" in ex and isinstance(ex["data"], list):
        turns = []
        for i, turn in enumerate(ex["data"]):
            if isinstance(turn, str) and turn.strip():
                role = "User" if i % 2 == 0 else "Assistant"
                turns.append(f"{role}: {turn.strip()}")
        return "\n".join(turns)
    return extract_text_from_example(ex)

def extract_text_from_example(ex):
    collected = []
    for k, v in ex.items():
        if isinstance(v, str) and v.strip():
            collected.append(v.strip())
        elif isinstance(v, (list, tuple)):
            sub = [str(x) for x in v if isinstance(x, str) and x.strip()]
            if sub:
                collected.append("\n".join(sub))
    if collected:
        return "\n".join(collected)
    return ""

def fetch_and_process(target_lines=TARGET_LINES):
    if load_dataset is None:
        raise RuntimeError("pip install datasets")

    print("→ Chargement du dataset 'stingning/ultrachat' en streaming...")
    ds = load_dataset("stingning/ultrachat", "default", split="train")

    # Vocabulaire temporaire
    chars_seen = set()

    file_index = 1
    current_file = get_file_path(file_index)
    fout = open(current_file, "w", encoding="utf-8")

    written = 0
    for ex in ds:
        if target_lines is not None and written >= target_lines:
            break

        conv_text = extract_full_conversation(ex)
        if not conv_text.strip():
            continue

        # Maj vocabulaire
        chars_seen.update(conv_text)

        # Écriture dans fichier actuel
        fout.write(conv_text + "\n\n")
        written += 1

        fout.flush()
        if os.path.getsize(current_file) >= MAX_FILE_SIZE:
            fout.close()
            print(f"💾 Fichier {current_file} plein (~2Go), passage au suivant...")
            file_index += 1
            current_file = get_file_path(file_index)
            fout = open(current_file, "w", encoding="utf-8")

        if written % 50000 == 0:
            print(f"  -> {written:,} conversations traitées...")

    fout.close()

    # Construire vocabulaire final
    chars_sorted = sorted(chars_seen)
    stoi = {ch: i for i, ch in enumerate(chars_sorted)}
    itos = {i: ch for ch, i in stoi.items()}

    # Sauvegarder meta
    meta = {"vocab_size": len(stoi), "itos": itos, "stoi": stoi}
    with open(META_PKL, "wb") as f:
        pickle.dump(meta, f)

    print(f"✔ Vocabulaire construit : {len(stoi)} caractères uniques")
    print(f"✔ Conversations totales : {written:,}")

    return file_index, stoi

def encode_files(num_files, stoi):
    """Lit chaque dataX.txt et encode directement en streaming vers train/val"""
    train_data = []
    val_data = []

    for i in range(1, num_files + 1):
        path = get_file_path(i)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            arr = np.array([stoi[c] for c in text], dtype=np.int32)

            # split 90% train / 10% val
            split_idx = int(len(arr) * 0.9)
            train_data.append(arr[:split_idx])
            val_data.append(arr[split_idx:])

    # Concaténer et sauvegarder
    train_data = np.concatenate(train_data)
    val_data = np.concatenate(val_data)

    with open(TRAIN_PKL, "wb") as f:
        pickle.dump(train_data, f)
    with open(VAL_PKL, "wb") as f:
        pickle.dump(val_data, f)

    print("\n✅ Sauvegardé :")
    print(f" - {META_PKL}")
    print(f" - {TRAIN_PKL} ({len(train_data):,} tokens)")
    print(f" - {VAL_PKL}   ({len(val_data):,} tokens)")

def main():
    num_files, stoi = fetch_and_process()
    encode_files(num_files, stoi)

if __name__ == "__main__":
    main()
