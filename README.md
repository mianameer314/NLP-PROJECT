# Urdu POS Tagger — BiLSTM + CRF (Deep Learning)

A Deep-Learning Part-of-Speech tagger for the **Urdu language** using a
Bidirectional LSTM with a CRF decoder, trained on the full
[UD Urdu Treebank](https://universaldependencies.org/treebanks/ur_udtb/index.html).

## Project Structure

```
NLP-PROJECT/
├── data/
│   └── sample_corpus.txt   ← 80 Urdu sentences (CLE format, for reference)
├── urdu_pos_tagger.ipynb   ← Main notebook: load → explore → train → evaluate → demo
└── README.md
```

## Quick Start

```bash
pip install torch pytorch-crf datasets scikit-learn matplotlib seaborn tqdm pandas
jupyter notebook urdu_pos_tagger.ipynb
```

## Dataset — UD Urdu Treebank

Loaded automatically from HuggingFace Hub (`universal-dependencies/universal_dependencies`, config `ur_udtb`).

| Split | Sentences | Tokens |
|-------|-----------|--------|
| Train | 4,323 | ~101 K |
| Dev   | 516   | ~12 K  |
| Test  | 535   | ~13 K  |

**Tags** — 17 Universal POS (UPOS) tags:

| Tag   | Meaning            | Tag    | Meaning            |
|-------|--------------------|--------|--------------------|
| NOUN  | Common Noun        | VERB   | Verb               |
| PROPN | Proper Noun        | AUX    | Auxiliary Verb     |
| PRON  | Pronoun            | ADJ    | Adjective          |
| DET   | Determiner         | ADV    | Adverb             |
| ADP   | Adposition/Postpos | NUM    | Numeral            |
| CCONJ | Coord. Conj.       | SCONJ  | Subord. Conj.      |
| PART  | Particle           | PUNCT  | Punctuation        |
| INTJ  | Interjection       | SYM    | Symbol             |
| X     | Other / Foreign    |        |                    |

## Model Architecture

```
Input tokens
      │
  Word Embeddings  (128-d, learned)
      │
  Dropout (0.3)
      │
  BiLSTM × 2 layers  (256-d hidden, bidirectional)
      │
  Linear (256 → num_tags)
      │
  CRF Decoder  ← models tag-transition constraints
      │
  Predicted tag sequence
```

The **CRF layer** replaces a plain softmax: it jointly decodes the whole sequence
and learns constraints such as "a VERB rarely follows PUNCT", yielding higher
accuracy than independent per-token classification.

## Notebook Sections

| # | Section | Content |
|---|---------|---------|
| 1 | Imports & Config | Hyper-parameters, device selection |
| 2 | Load Dataset | UD Urdu Treebank via HuggingFace `datasets` |
| 3 | Data Exploration | Token stats, tag frequency & length distribution plots |
| 4 | Preprocessing | Word vocabulary, PyTorch `Dataset` & `DataLoader` |
| 5 | Model Architecture | `BiLSTM_CRF` class definition |
| 6 | Training | Adam + LR scheduler, loss/accuracy curves |
| 7 | Evaluation | Classification report + normalised confusion matrix |
| 8 | Error Analysis | Most-confused tag pairs, per-tag error rates |
| 9 | Live Demo | Tag arbitrary Urdu sentences + colour-coded table |

## Expected Results

- **Test accuracy**: ≥ 90 %
- **Macro F1**: ≥ 0.88
