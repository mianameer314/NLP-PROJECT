# Urdu POS Tagger — HMM with NLTK

A minimal Part-of-Speech tagger for the **Urdu language** using NLTK's built-in Hidden Markov Model (HMM) trainer and Viterbi decoding.

## Project Structure

```
NLP-PROJECT/
├── data/
│   └── sample_corpus.txt   ← 80 Urdu sentences (CLE format: WORD\tTAG)
├── urdu_pos_tagger.ipynb   ← Single notebook: load → train → evaluate → demo
└── README.md
```

## Quick Start

```bash
pip install nltk scikit-learn matplotlib seaborn
jupyter notebook urdu_pos_tagger.ipynb
```

## Data Format (CLE Style)

Each line contains `WORD<TAB>TAG`. A blank line marks the end of a sentence.

```
میں	PRP
نے	IN
کتاب	NN
پڑھی	VBD
۔	PUNC
```

## POS Tags

| Tag  | Meaning              | Example        |
|------|----------------------|----------------|
| NN   | Common Noun          | کتاب (book)    |
| NNP  | Proper Noun          | لاہور (Lahore) |
| PRP  | Pronoun              | میں (I/me)     |
| VBZ  | Present Verb         | ہے (is)        |
| VBD  | Past Verb            | پڑھی (read)    |
| VB   | Base Verb            | پڑھ (read)     |
| JJ   | Adjective            | اچھا (good)    |
| RB   | Adverb               | بہت (very)     |
| IN   | Postposition         | میں (in)       |
| DT   | Determiner           | یہ (this)      |
| CC   | Conjunction          | اور (and)      |
| CD   | Cardinal Number      | دو (two)       |
| NEG  | Negation             | نہیں (not)     |
| WP   | Question Word        | کیا (what)     |
| PUNC | Punctuation          | ۔ ؟            |

## Algorithm

The notebook uses **NLTK's `HiddenMarkovModelTrainer`** — a supervised HMM with:
- Transition probabilities learned from training data
- Emission probabilities with Laplace smoothing
- Viterbi decoding for optimal tag sequence inference
