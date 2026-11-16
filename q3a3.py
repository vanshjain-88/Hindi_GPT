"""
Task 3
REQUIRED FILES FOR RUNNING:
    ./tokenizer/tokenizer.json -> SAVED BY TASK 1
    ./text_classification_dataset/train.csv   -> GIVEN
    ./best_model.pt            ->  SAVED BY TASK -2
"""

import math
import os
import zipfile
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Device ───────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# ── Tokenizer ────────────────────────────────────────────────────
tok        = Tokenizer.from_file("./tokenizer/tokenizer.json")
VOCAB_SIZE = tok.get_vocab_size()
EOS_ID     = tok.token_to_id("<|eos|>")
PAD_ID     = tok.token_to_id("<|pad|>")
print(f"Vocab size: {VOCAB_SIZE}")


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = VOCAB_SIZE
    n_layer:    int = 4
    n_head:     int = 4
    n_embd:     int = 128


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                  .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv     = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y   = att @ v
        y   = y.transpose(1, 2).contiguous().view(B, T, C)
        y   = self.c_proj(y)
        return y


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.relu   = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1       = nn.LayerNorm(config.n_embd)
        self.attention = MultiHeadAttention(config)
        self.ln2       = nn.LayerNorm(config.n_embd)
        self.ffn       = FFN(config)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTBaseWithHead(nn.Module):
    """
    Task-2 model — kept identical so state_dict keys match best_model.pt.
    encode() returns hidden states (B, T, n_embd) without the LM head.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x    = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def encode(self, idx):
        """Returns final hidden states (B, T, n_embd) — no LM head."""
        B, T = idx.size()
        pos  = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x    = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)   # (B, T, n_embd)


# ════════════════════════════════════════════════════════════════
#  Task 3: GPTClassifier
# ════════════════════════════════════════════════════════════════

NUM_CLASSES = 3

class GPTClassifier(nn.Module):
    """
    GPTBaseWithHead backbone + classification head.

    Sequence representation: hidden state of the last real token.
    Because this is a causal (left-to-right) model, the last token
    has attended to all previous tokens — it carries the most
    accumulated context and is the natural summary of the sequence.

    Dropout(0.3) before cls_head regularises the small classification
    dataset and prevents overfitting.
    """
    def __init__(self, config, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = GPTBaseWithHead(config)
        self.dropout  = nn.Dropout(0.3)
        self.cls_head = nn.Linear(config.n_embd, num_classes, bias=True)
        nn.init.normal_(self.cls_head.weight, std=0.02)
        nn.init.zeros_(self.cls_head.bias)

    def forward(self, idx):
        """
        idx : (B, T) token ids, right-padded with PAD_ID

        Returns logits (B, num_classes).
        """
        x = self.backbone.encode(idx)      # (B, T, n_embd)

        # Extract the last token's hidden state as the sequence representation.
        # In a causal model each token can only attend to tokens before it,
        # so the final token position has seen the entire sequence — it is
        # the richest single-vector summary available.
        features = x[:, -1, :]             # (B, n_embd)

        features = self.dropout(features)
        return self.cls_head(features)     # (B, num_classes)



def load_pretrained_backbone(classifier: GPTClassifier,
                             ckpt_path: str = "./best_model.pt"):
    if not os.path.exists(ckpt_path):
        print(f"[WARNING] '{ckpt_path}' not found — training from scratch.")
        return

    if os.path.isdir(ckpt_path):
        print(f"  (detected folder-format checkpoint — re-zipping for torch.load)")
        buf         = io.BytesIO()
        folder_name = os.path.basename(ckpt_path)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for root, _, files in os.walk(ckpt_path):
                for fname in sorted(files):
                    full    = os.path.join(root, fname)
                    arcname = os.path.join(
                        folder_name,
                        os.path.relpath(full, ckpt_path)
                    )
                    zinfo = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                    zinfo.compress_type = zipfile.ZIP_STORED
                    with open(full, "rb") as fh:
                        zf.writestr(zinfo, fh.read())
        buf.seek(0)
        state = torch.load(buf, map_location=device, weights_only=False)
    else:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefix keys with "backbone." to match GPTClassifier's structure
    backbone_state = {"backbone." + k: v for k, v in state.items()}
    missing, unexpected = classifier.load_state_dict(backbone_state, strict=False)

    cls_missing  = [k for k in missing if "cls_head" in k or "dropout" in k]
    real_missing = [k for k in missing if k not in cls_missing]

    print(f"Loaded pretrained backbone from '{ckpt_path}'")
    if real_missing:
        print(f"  Truly missing keys : {real_missing}")
    if unexpected:
        print(f"  Unexpected keys    : {unexpected}")
    print(f"  Head initialised fresh (expected): {cls_missing}")


# ════════════════════════════════════════════════════════════════
#  Dataset & DataLoader
# ════════════════════════════════════════════════════════════════

class ReviewDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def collate_fn(batch):
    xs, ys    = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=PAD_ID)
    xs_padded = xs_padded[:, :config.block_size]
    ys        = torch.stack(list(ys))
    return xs_padded, ys


def build_dataloaders(batch_size=32):
    df = pd.read_csv("./text_classification_dataset/train.csv")
    print(f"Dataset: {df.shape}  |  columns: {list(df.columns)}")
    print(df.head(3))

    unique_labels = sorted(df["experience"].unique())
    label_map     = {v: i for i, v in enumerate(unique_labels)}
    print(f"Label map: {label_map}")
    df["label"]   = df["experience"].map(label_map)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)

    def tokenize_df(split_df):
        X, Y = [], []
        for _, row in split_df.iterrows():
            ids = tok.encode(str(row["text"])).ids + [EOS_ID]
            # Tail truncation: keep last block_size tokens.
            # Sentiment conclusions appear near the end of a review.
            ids = ids[-config.block_size:]
            X.append(torch.tensor(ids, dtype=torch.long))
            Y.append(torch.tensor(int(row["label"]), dtype=torch.long))
        return X, Y

    X_train, y_train = tokenize_df(train_df)
    X_val,   y_val   = tokenize_df(val_df)

    train_loader = DataLoader(ReviewDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(ReviewDataset(X_val, y_val),
                              batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    return train_loader, val_loader, val_df, label_map


# ════════════════════════════════════════════════════════════════
#  Train / Evaluate
# ════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(dim=-1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(dim=-1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total


# ════════════════════════════════════════════════════════════════
#  Print 5 correctly predicted reviews
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def show_correct_predictions(model, val_df, label_map, n=5):
    inv_map    = {v: k for k, v in label_map.items()}
    label_name = {0: "Negative", 1: "Neutral", 2: "Positive"}

    model.eval()
    shown = 0
    print(f"\n{'─'*65}")
    print(f"  {n} Correctly Predicted Validation Reviews")
    print(f"{'─'*65}")

    for _, row in val_df.iterrows():
        if shown >= n:
            break
        ids = tok.encode(str(row["text"])).ids + [EOS_ID]
        ids = ids[-config.block_size:]           # tail truncation
        x   = torch.tensor([ids], dtype=torch.long, device=device)

        pred = model(x).argmax(dim=-1).item()
        true = int(row["label"])

        if pred == true:
            name = label_name.get(true, str(inv_map.get(true, true)))
            print(f"\n[{shown+1}] True  : {name} (class {true})")
            print(f"     Pred  : {name} (class {pred})  ✓")
            print(f"     Text  : {str(row['text'])[:220]}")
            shown += 1

    if shown < n:
        print(f"\n(Only {shown} correct examples found in validation set.)")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

BATCH_SIZE = 32
MAX_EPOCHS = 50
LR         = 5e-4

CLASSIFIER_CKPT = "./best_classifier.pt"

config = GPTConfig()

# val_df and label_map are always needed (for show_correct_predictions)
_, val_loader, val_df, label_map = build_dataloaders(BATCH_SIZE)

# ── Build model ───────────────────────────────────────────────────
model = GPTClassifier(config, num_classes=NUM_CLASSES).to(device)

# ════════════════════════════════════════════════════════════════
#  BRANCH: skip training if best_classifier.pt already exists
# ════════════════════════════════════════════════════════════════
if os.path.exists(CLASSIFIER_CKPT):
    print(f"\n[INFO] Found '{CLASSIFIER_CKPT}' — skipping training.")
    model.load_state_dict(
        torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)
    )
    print("[INFO] Classifier weights loaded successfully.")

else:
    print(f"\n[INFO] '{CLASSIFIER_CKPT}' not found — starting training.")

    train_loader, _, _, _ = build_dataloaders(BATCH_SIZE)

    load_pretrained_backbone(model, "./best_model.pt")


    for p in model.backbone.parameters():
        p.requires_grad = False
    for block in model.backbone.transformer.h[-2:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.backbone.transformer.ln_f.parameters():
        p.requires_grad = True
    # cls_head and dropout are new layers — always trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}  (cls_head + last 2 blocks + ln_f)")

    # Class-weighted loss to handle imbalanced label distribution
    train_labels = []
    for _, ys in train_loader:
        train_labels.extend(ys.tolist())
    counts  = torch.zeros(NUM_CLASSES)
    for lbl in train_labels:
        counts[lbl] += 1
    weights   = (counts.sum() / (NUM_CLASSES * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class counts : {counts.tolist()}")
    print(f"Class weights: {[round(w, 4) for w in weights.tolist()]}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        scheduler.step(vl_acc)

        print(f"Epoch {epoch:2d}/{MAX_EPOCHS} | "
              f"train loss {tr_loss:.4f}  acc {tr_acc*100:.2f}% | "
              f"val loss {vl_loss:.4f}  acc {vl_acc*100:.2f}%  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), CLASSIFIER_CKPT)
            print(f"           ↳ New best val acc {best_val_acc*100:.2f}%"
                  f"  (saved to best_classifier.pt)")

    # Plots — only generated after a fresh training run
    epochs_range = range(1, MAX_EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, train_losses, label="train loss")
    ax1.plot(epochs_range, val_losses,   label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax1.set_title("Classification Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs_range, [a * 100 for a in train_accs], label="train acc")
    ax2.plot(epochs_range, [a * 100 for a in val_accs],   label="val acc")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Classification Accuracy"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("classification_curves.png", dpi=150)
    plt.show()
    print("Plot saved to classification_curves.png")

    # Reload best checkpoint before final evaluation
    model.load_state_dict(
        torch.load(CLASSIFIER_CKPT, map_location=device, weights_only=False)
    )



criterion = nn.CrossEntropyLoss()

_, final_acc = evaluate(model, val_loader, criterion)

sep = "═" * 65
print(f"\n{sep}")
print(f"  Final validation accuracy : {final_acc*100:.2f}%")
print(sep)

show_correct_predictions(model, val_df, label_map, n=5)