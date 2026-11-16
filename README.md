# Hindi GPT

A decoder-only GPT language model trained from scratch on a Hindi corpus, with a custom BPE tokenizer and a downstream three-class review classifier.

## Highlights

- 19.7M-parameter GPT with causal multi-head self-attention and tied token/output embeddings
- 5,000-token Hindi BPE vocabulary trained on 41,800 documents
- 41.50 final validation perplexity after 5,000 language-model training steps
- Transfer-learning classifier with class-weighted loss and selective transformer fine-tuning

## Pipeline

1. `NLP_A3.ipynb` / `q1a3.py` trains the tokenizer and encodes the Hindi corpus.
2. `A3_Q2.ipynb` trains and evaluates the autoregressive language model.
3. `q3a3.py` adapts the pretrained backbone for three-class review classification.

## Repository contents

```text
.
├── NLP_A3.ipynb
├── A3_Q2.ipynb
├── q1a3.py
├── q3a3.py
├── tokenizer/
│   ├── tokenizer.json
│   └── special_tokens.txt
├── q2_training_curves.png
├── predicted_reviews.txt
└── requirements.txt
```

Raw datasets, encoded binaries, archives, and model checkpoints are intentionally excluded. Place the Hindi corpus in `hindi_corpus/train/`, classification data in `text_classification_dataset/train.csv`, and the pretrained checkpoint at `best_model.pt` before running the full pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in order for tokenizer and language-model training, then execute:

```bash
python q3a3.py
```

## Resume highlights

- Built 19.7M-parameter Hindi GPT with 10 transformer layers, reaching 41.50 validation perplexity after 5,000 steps.
- Engineered 5,000-token BPE pipeline processing 41,800 Hindi documents with reproducible 95/5 training-validation data splits efficiently.
- Fine-tuned 2 transformer blocks for 3-class review classification using weighted loss and 128-token contexts effectively.
