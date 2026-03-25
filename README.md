# Urdu HMM Part-of-Speech Tagger

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **Hidden Markov Model (HMM)** for Part-of-Speech (POS) tagging on Urdu text, implemented from scratch in Python. Uses the **Viterbi algorithm** for decoding and achieves **~93% token-level accuracy** on the included corpus.

> **Author:** Aashan Khan | **Course:** Machine Learning / Classical NLP  
> **References:** [CLE Urdu POS Tagset (LREC 2014)](http://www.lrec-conf.org/proceedings/lrec2014/pdf/275_Paper.pdf) · [Anwar et al. (2007), HMM-based Urdu POS Tagger](https://pdfs.semanticscholar.org/7a3f/1ea18bf3e8223890b122bc31fb79db758c6e.pdf) · [UD Urdu Treebank](https://universaldependencies.org/treebanks/ur_udtb/index.html)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Dataset](#dataset)
5. [Methodology](#methodology)
6. [Results](#results)
7. [Viterbi Visualization](#viterbi-visualization)
8. [Usage](#usage)
9. [References](#references)

---

## Project Overview

This project implements a classical NLP pipeline for **Urdu POS tagging**:

| Component | Details |
|-----------|---------|
| Model | First-order Hidden Markov Model (HMM) |
| Training | MLE + Laplace (add-k) smoothing |
| Decoding | Viterbi dynamic programming |
| Language | Urdu (UTF-8, Nastaliq script, right-to-left) |
| Tagset | 14 POS tags (CLE-style) |
| Accuracy | ~93% token-level on sample corpus |

---

## Repository Structure

```
NLP-PROJECT/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
│
├── data/
│   ├── README.md                    ← Data sources & format description
│   ├── sample_corpus.txt            ← Full synthetic corpus (400 sentences)
│   ├── train.txt                    ← Training split (320 sentences, 80%)
│   └── test.txt                     ← Test split    (80 sentences,  20%)
│
├── src/
│   ├── __init__.py                  ← Package exports
│   ├── data_loader.py               ← CLE & CoNLL-U format parsers
│   ├── hmm_tagger.py                ← HMM model + Viterbi algorithm
│   └── evaluate.py                  ← Accuracy, per-tag F1, confusion matrix
│
└── notebooks/
    └── urdu_hmm_pos_tagger.ipynb    ← Full analysis notebook
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
```bash
cd notebooks
jupyter notebook urdu_hmm_pos_tagger.ipynb
```

### 3. Or use the Python API directly
```python
from src import DataLoader, HMMTagger, evaluate, print_report

# Load data
loader = DataLoader(format='cle')
train_sents = loader.load('data/train.txt')
test_sents  = loader.load('data/test.txt')

# Train
tagger = HMMTagger(smoothing_k=1.0)
tagger.train(train_sents)

# Tag a sentence
words = 'میں نے کتاب پڑھا ۔'.split()
tags  = tagger.tag(words)
print(list(zip(words, tags)))
# [('میں', 'PRP'), ('نے', 'IN'), ('کتاب', 'NN'), ('پڑھا', 'VBD'), ('۔', 'PUNC')]

# Evaluate
results = evaluate(tagger, test_sents)
print_report(results)
```

---

## Dataset

### Included Corpus (Sample)
- **400 sentences**, ~2,300 tokens in CLE-style tab-separated format
- Covers diverse sentence structures: declarative, interrogative, negative
- 14 POS tags, 200-word vocabulary

### Recommended: Real Datasets

#### CLE Urdu POS Tagged Corpus _(Primary)_
> Sajjad, H., & Schmid, H. (2009). Tagging Urdu Text with Parts of Speech: A Tagger Comparison. EACL.  
> Hardie, A. (2003). The CLE Urdu POS Tagset.

- **Size**: ~100,000 words, **32 fine-grained POS tags**
- **Source**: [Center for Language Engineering (CLE)](http://www.cle.org.pk/software/ling_resources/UrduPOSCorpus.htm), UET Lahore
- **Format**: CLE-style (word\tTAG, blank lines between sentences)
- **Reference**: [CLE Urdu POS Tagset paper (LREC 2014)](http://www.lrec-conf.org/proceedings/lrec2014/pdf/275_Paper.pdf)

#### Universal Dependencies Urdu Treebank _(Alternative)_
- **Size**: ~138,000 tokens, **16 UPOS tags**
- **Source**: [UD Urdu Treebank](https://universaldependencies.org/treebanks/ur_udtb/index.html)
- **Format**: CoNLL-U
- **Download**:
  ```bash
  git clone https://github.com/UniversalDependencies/UD_Urdu-UDTB.git
  ```
- **Load with this project**:
  ```python
  loader = DataLoader(format='conllu')
  train = loader.load('UD_Urdu-UDTB/ur_udtb-ud-train.conllu')
  ```

---

## Methodology

### 1. Preprocessing
- Parse word/tag pairs from corpus files
- Normalize Urdu Unicode (strip zero-width characters)
- Split 80/20 into train/test sets

### 2. HMM Training (Maximum Likelihood Estimation)

**Transition probabilities** — how likely a tag follows another:

$$P(t_i \mid t_{i-1}) = \frac{C(t_{i-1},\, t_i) + k}{C(t_{i-1}) + k \cdot |\text{Tags}|}$$

**Emission probabilities** — how likely a word is given its tag:

$$P(w_i \mid t_i) = \frac{C(t_i,\, w_i) + k}{C(t_i) + k \cdot |V|}$$

Both probabilities are smoothed with **Laplace (add-k) smoothing** (default `k=1.0`) to handle unseen word/tag combinations.

### 3. Viterbi Decoding

The Viterbi algorithm finds the most probable tag sequence in O(n·T²) time:

$$V(t, k) = \max_j \left[ V(t-1,\, j) \cdot P(k \mid j) \cdot P(w_t \mid k) \right]$$

Computed in **log-space** to prevent numerical underflow:

$$\log V(t, k) = \max_j \left[ \log V(t-1,\, j) + \log P(k \mid j) + \log P(w_t \mid k) \right]$$

The best path is recovered via backtracking through the trellis.

### 4. Unknown Word Handling
OOV words fall back to UNK emission probabilities, adjusted by **Urdu-specific suffix heuristics**:
- Past-tense verb endings (e.g., words ending in `ا`, `یا`)
- Punctuation marks (`۔`, `،`, `؟`)
- Adjective suffixes (`ا`, `ے`, `ی`)

---

## Results

| Metric | Value |
|--------|-------|
| Overall accuracy | **93.2%** |
| In-vocabulary accuracy | 93.9% |
| OOV accuracy | 33.3% |

**Per-tag F1 scores:**

| Tag | Description | Precision | Recall | F1 |
|-----|-------------|-----------|--------|----|
| NEG | Negation | 1.000 | 1.000 | **1.000** |
| VBD | Past Verb | 0.971 | 1.000 | **0.985** |
| PUNC | Punctuation | 0.975 | 1.000 | **0.987** |
| NN | Common Noun | 0.979 | 0.969 | **0.974** |
| VBZ | Present Verb | 1.000 | 0.900 | **0.947** |
| PRP | Pronoun | 0.840 | 0.984 | 0.906 |
| IN | Postposition | 0.827 | 1.000 | 0.905 |
| JJ | Adjective | 1.000 | 0.750 | 0.857 |

---

## Viterbi Visualization

The notebook generates several visualizations:

### Tag Sequence Color Map
Each sentence is displayed as colored word boxes with POS tags:

| Color | Tag | Meaning |
|-------|-----|---------|
| 🔵 Blue | NN / NNP | Nouns |
| 🟣 Purple | PRP | Pronouns |
| 🔴 Red/Orange | VB / VBD / VBZ | Verbs |
| 🟢 Green | JJ | Adjectives |
| 🩵 Teal | RB | Adverbs |
| 🟡 Gold | DT | Determiners |
| ⬜ Gray | IN / CC / PUNC | Postpositions, Conjunctions, Punctuation |

### Viterbi Trellis
The trellis diagram shows log-probability scores at each (time-step, tag) node, with the best path highlighted in green and backpointers shown as red arrows.

### Other Plots
- POS tag frequency distribution
- HMM transition probability matrix (heatmap)
- Per-tag Precision/Recall/F1 bar chart
- Confusion matrix (counts + normalized)
- Smoothing parameter sensitivity curve

---

## Usage

### DataLoader
```python
from src import DataLoader

# CLE format (default)
loader = DataLoader(format='cle')
sentences = loader.load('data/train.txt')

# CoNLL-U format (Universal Dependencies)
loader = DataLoader(format='conllu')
sentences = loader.load('path/to/ud_file.conllu')

# Print stats
print(loader.corpus_stats(sentences))
```

### HMMTagger
```python
from src import HMMTagger

tagger = HMMTagger(smoothing_k=1.0)
tagger.train(train_sentences)

# Tag a list of words
tags = tagger.tag(['وہ', 'لاہور', 'میں', 'رہتا', 'ہے', '۔'])

# Tag a string (whitespace-tokenised)
result = tagger.tag_sentence_str('وہ لاہور میں رہتا ہے ۔')
# [('وہ', 'PRP'), ('لاہور', 'NNP'), ('میں', 'IN'), ('رہتا', 'VBZ'), ('ہے', 'VBZ'), ('۔', 'PUNC')]

# Viterbi with log-probability
best_tags, log_prob = tagger.viterbi(words)

# Inspect model
print(tagger.get_top_emissions('NN', n=5))
print(tagger.get_top_transitions('PRP', n=5))
```

### Evaluate
```python
from src import evaluate, print_report, error_analysis

results = evaluate(tagger, test_sentences)
print_report(results)

# Error analysis
errors = error_analysis(tagger, test_sentences, n=20)
```

---

## References

1. Hardie, A. (2003). *Developing a tagset for automated part-of-speech tagging in Urdu*. CORPUS Linguistics. [semanticscholar](https://www.semanticscholar.org/paper/Developing-a-tagset-for-automated-part-of-speech-in-Hardie/9a2b14ea8432e5e8ccb73c87c329fd24c4489575)

2. Sajjad, H., & Schmid, H. (2009). *Tagging Urdu Text with Parts of Speech: A Tagger Comparison*. EACL 2009. [PDF](http://www.lrec-conf.org/proceedings/lrec2014/pdf/275_Paper.pdf)

3. Anwar, W., Wang, X., & Chen, X. (2007). *HMM-based Urdu POS Tagger*. Information Technology Journal.  [PDF](https://pdfs.semanticscholar.org/7a3f/1ea18bf3e8223890b122bc31fb79db758c6e.pdf)

4. Jurafsky, D. & Martin, J.H. (2023). *Speech and Language Processing*, 3rd ed., Ch. 8 (HMM & Viterbi). [online draft](https://web.stanford.edu/~jurafsky/slp3/)

5. Universal Dependencies Urdu Treebank. [universaldependencies.org](https://universaldependencies.org/treebanks/ur_udtb/index.html)

6. CLE Urdu POS Tagset paper (LREC 2014). [lrec-conf.org](http://www.lrec-conf.org/proceedings/lrec2014/pdf/275_Paper.pdf)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
