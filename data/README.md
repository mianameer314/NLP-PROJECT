# Urdu HMM POS Tagger — Data Directory

This directory contains the corpus data used for training and evaluating the HMM POS tagger.

## Files

| File | Description |
|------|-------------|
| `sample_corpus.txt` | Full synthetic Urdu POS corpus (400 sentences, ~2,300 tokens) |
| `train.txt` | Training split (80% — 320 sentences) |
| `test.txt` | Test split (20% — 80 sentences) |

## Format

Each file uses the **CLE-style** tab-separated format:

```
word\tPOS_TAG
word\tPOS_TAG
...
            ← blank line separates sentences
word\tPOS_TAG
```

Example:
```
میں	PRP
نے	IN
کتاب	NN
پڑھا	VBD
۔	PUNC
```

## POS Tagset (13 Tags)

| Tag | Description | Example (Urdu) | Gloss |
|-----|-------------|----------------|-------|
| `NN` | Common Noun | کتاب | book |
| `NNP` | Proper Noun | لاہور | Lahore |
| `PRP` | Pronoun | میں، وہ | I, he/she |
| `VB` | Verb (base) | کرنا | to do |
| `VBZ` | Verb (present) | ہے، ہیں | is, are |
| `VBD` | Verb (past) | کیا، گیا | did, went |
| `JJ` | Adjective | اچھا، بڑا | good, big |
| `RB` | Adverb | بہت، اب | very, now |
| `DT` | Determiner | یہ، وہ | this, that |
| `CC` | Conjunction | اور، لیکن | and, but |
| `IN` | Postposition | میں، پر، کا | in, on, of |
| `CD` | Cardinal Number | ایک، دو | one, two |
| `NEG` | Negation | نہیں، نہ | not |
| `WP` | Question Word | کیا، کون | what, who |
| `PUNC` | Punctuation | ۔، ، | period, comma |

## Using the Real Dataset (Recommended for Research)

### Option 1: CLE Urdu POS Tagged Corpus
- **Source**: [Center for Language Engineering (CLE), UET Lahore](http://www.cle.org.pk/software/ling_resources/UrduPOSCorpus.htm)
- **Size**: ~100,000 words, 32 fine-grained POS tags
- **Script**: UTF-8 Urdu
- **Format**: Tab-separated word/tag pairs
- **Reference**: Sajjad & Schmid (2009). Tagging Urdu Text with Parts of Speech. EACL.

After downloading, place files as:
```
data/
  cle_train.txt
  cle_test.txt
```

### Option 2: Universal Dependencies Urdu Treebank (UD)
- **Source**: [UD Urdu Treebank](https://universaldependencies.org/treebanks/ur_udtb/index.html)
- **Size**: ~138,000 tokens, 16 UPOS tags
- **Format**: CoNLL-U
- **Direct download**:
  ```bash
  # Via UD GitHub
  git clone https://github.com/UniversalDependencies/UD_Urdu-UDTB.git
  ```

After downloading, use the `--format conllu` flag with the tagger:
```python
loader = DataLoader(format='conllu')
sentences = loader.load('UD_Urdu-UDTB/ur_udtb-ud-train.conllu')
```

## Notes on Urdu Script
- Urdu is written **right-to-left** in Nastaliq script
- All files use **UTF-8** encoding
- Punctuation includes Urdu-specific characters: `۔` (period), `،` (comma), `؟` (question mark)
