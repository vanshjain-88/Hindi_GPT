import os
import zipfile
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
import numpy as np
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


DATA_DIR = "./hindi_corpus/train"

def load_hindi_corpus():
    texts = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(DATA_DIR, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts


all_texts = load_hindi_corpus()

train_texts, val_texts = train_test_split(
    all_texts, test_size=0.1, random_state=42
)

PROJECT_DIR = "./"
TOK_DIR = os.path.join(PROJECT_DIR, "tokenizer")
os.makedirs(TOK_DIR, exist_ok=True)

# DATASET_NAME = "Skylion007/openwebtext"
# Dataset_dir="content/"

# RAW_TEXT_BUDGET_BYTES = 3 * 1024**3   # 3GB (set 2*1024**3 for 2GB)
VOCAB_SIZE = 5000
# MIN_FREQUENCY = 2

PAD = "<|pad|>"
UNK = "<|unk|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]


def train_iterator():
    for text in train_texts:
        yield text

tokenizer = Tokenizer(BPE(unk_token=UNK))
tokenizer.normalizer = NFKC()
# tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
# tokenizer.decoder = ByteLevelDecoder()

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    # min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
)

# tokenizer.train_from_iterator(train_iterator(), trainer=trainer)
tokenizer.train_from_iterator(
    tqdm(train_iterator(), total=len(train_texts), desc="Training Tokenizer"),
    trainer=trainer
)

tok_path = os.path.join(TOK_DIR, "tokenizer.json")
tokenizer.save(tok_path)

vocab = tokenizer.get_vocab()
with open(os.path.join(TOK_DIR, "special_tokens.txt"), "w", encoding="utf-8") as f:
    for t in SPECIAL_TOKENS:
        f.write(f"{t}\t{vocab[t]}\n")

print("Saved tokenizer:", tok_path)
print("Vocab size:", tokenizer.get_vocab_size())
print("EOS id:", vocab[EOS])


TOK_PATH = "./tokenizer/tokenizer.json"

tokenizer = Tokenizer.from_file(TOK_PATH)
vocab=tokenizer.get_vocab()
print("Loaded tokenizer.")
print("Vocab size:", tokenizer.get_vocab_size())
# print(f"फिल्म->{vocab["फ्रांस"]}")
print(vocab)

result=tokenizer.encode("फ्रांसीसी क्रांति फ्रांस के इतिहास की राजनैतिक")
print(result.tokens)
print(result.ids)

PROJECT_DIR = "./"
TOK_PATH = os.path.join(PROJECT_DIR, "tokenizer", "tokenizer.json")

OUT_DIR = os.path.join(PROJECT_DIR, "data", "encoded")
os.makedirs(OUT_DIR, exist_ok=True)

tok = Tokenizer.from_file(TOK_PATH)
vocab = tok.get_vocab()
EOS_ID = vocab["<|eos|>"]
def tokenize_and_save(texts, out_path):
    token_buffer = []

    for text in tqdm(texts):
        ids = tok.encode(text).ids
        ids.append(EOS_ID)
        token_buffer.extend(ids)

    arr = np.array(token_buffer, dtype=np.uint32)
    arr.tofile(out_path)

train_path = os.path.join(OUT_DIR, "train.bin")
val_path = os.path.join(OUT_DIR, "val.bin")

tokenize_and_save(train_texts, train_path)
tokenize_and_save(val_texts, val_path)

DATA_DIR = "./data/encoded"
train_data = np.memmap(f"{DATA_DIR}/train.bin", dtype=np.uint32, mode="r")
val_data   = np.memmap(f"{DATA_DIR}/val.bin", dtype=np.uint32, mode="r")
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split, batch_size=4, block_size=128, device=device):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# quick sanity check
x, y = get_batch("train", batch_size=4, block_size=128, device=device)
print(x.shape, y.shape, x.dtype, y.dtype, "device:", x.device)

x2,y2=get_batch("train", batch_size=4,block_size=128,device=device)
print("x:")
print(x)
print("x2:")
print(x2)

def decode(ids):
    return tok.decode(ids)

print(decode([1478, 1448, 1459, 1601]))

# zip_path = '/content/drive/My Drive/text_classification_dataset.zip'
# extract_path = './'

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)

CLS_PATH = "./text_classification_dataset/train.csv"

df = pd.read_csv(CLS_PATH)
print(df.head())

train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42
)

tok = Tokenizer.from_file(TOK_PATH)

EOS_ID = tok.token_to_id("<|eos|>")

def process_classification(df):
    X, Y = [], []

    for _, row in df.iterrows():
        text = str(row["text"])
        label = int(row["experience"])

        ids = tok.encode(text).ids
        ids.append(EOS_ID)

        X.append(ids)
        Y.append(label)

    return X, Y


X_train, y_train = process_classification(train_df)
X_val, y_val = process_classification(val_df)


X_train = [torch.tensor(x, dtype=torch.long) for x in X_train]
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = [torch.tensor(x, dtype=torch.long) for x in X_val]
y_val = torch.tensor(y_val, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)

    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.tensor(ys)

    return xs_padded, ys


class ReviewDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_dataset = ReviewDataset(X_train, y_train)
val_dataset = ReviewDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

# print(train_loader)


batch = next(iter(train_loader))
x, y = batch

print(x)
print(y)