"""
Urdu HMM POS Tagger — Data Loader
Supports CLE-style tab-separated format and CoNLL-U format.
"""
import os
import re
from typing import List, Tuple, Dict


Sentence = List[Tuple[str, str]]   # [(word, tag), ...]


class DataLoader:
    """Load and preprocess Urdu POS-tagged corpora.

    Supported formats:
        'cle'     — CLE-style: word<TAB>tag, blank lines between sentences
        'conllu'  — Universal Dependencies CoNLL-U format
    """

    def __init__(self, format: str = 'cle'):
        if format not in ('cle', 'conllu'):
            raise ValueError("format must be 'cle' or 'conllu'")
        self.format = format

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> List[Sentence]:
        """Load a corpus file and return a list of sentences."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        if self.format == 'cle':
            return self._parse_cle(text)
        return self._parse_conllu(text)

    def load_train_test(
        self,
        train_path: str,
        test_path: str,
    ) -> Tuple[List[Sentence], List[Sentence]]:
        """Convenience wrapper to load both splits at once."""
        return self.load(train_path), self.load(test_path)

    def train_test_split(
        self,
        sentences: List[Sentence],
        test_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Sentence], List[Sentence]]:
        """Split a single corpus into train/test sets."""
        import random
        data = list(sentences)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(data)
        split = int(len(data) * (1 - test_ratio))
        return data[:split], data[split:]

    # ------------------------------------------------------------------
    # Corpus statistics
    # ------------------------------------------------------------------

    def corpus_stats(self, sentences: List[Sentence]) -> Dict:
        """Return basic statistics about a corpus."""
        tokens = [pair for sent in sentences for pair in sent]
        words = [w for w, _ in tokens]
        tags = [t for _, t in tokens]
        tag_freq: Dict[str, int] = {}
        for t in tags:
            tag_freq[t] = tag_freq.get(t, 0) + 1

        return {
            'num_sentences': len(sentences),
            'num_tokens': len(tokens),
            'vocab_size': len(set(words)),
            'num_tags': len(set(tags)),
            'tag_freq': dict(sorted(tag_freq.items(), key=lambda x: -x[1])),
        }

    # ------------------------------------------------------------------
    # Parsers (private)
    # ------------------------------------------------------------------

    def _parse_cle(self, text: str) -> List[Sentence]:
        """Parse CLE-style corpus (word\\ttag, blank lines = sentence boundaries)."""
        sentences: List[Sentence] = []
        current: Sentence = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                # Try whitespace split as fallback
                parts = line.split()
            if len(parts) >= 2:
                word = self._normalize(parts[0])
                tag = parts[1].strip()
                if word and tag:
                    current.append((word, tag))
        if current:
            sentences.append(current)
        return sentences

    def _parse_conllu(self, text: str) -> List[Sentence]:
        """Parse Universal Dependencies CoNLL-U format.

        Uses column 1 (FORM) and column 3 (UPOS).
        Multi-word tokens (lines with '-' in ID) and empty nodes ('.' in ID)
        are skipped.
        """
        sentences: List[Sentence] = []
        current: Sentence = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith('#'):
                continue
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            cols = line.split('\t')
            if len(cols) < 4:
                continue
            token_id = cols[0]
            # Skip multi-word and empty tokens
            if '-' in token_id or '.' in token_id:
                continue
            word = self._normalize(cols[1])
            upos = cols[3].strip()
            if word and upos and upos != '_':
                current.append((word, upos))
        if current:
            sentences.append(current)
        return sentences

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(word: str) -> str:
        """Minimal Unicode normalization for Urdu text."""
        # Strip zero-width characters and normalize whitespace
        word = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', word)
        return word.strip()
