"""
Urdu HMM POS Tagger
===================
Implements a first-order Hidden Markov Model for Part-of-Speech tagging with:
  - Maximum Likelihood Estimation + Laplace (add-k) smoothing
  - Viterbi dynamic-programming decoding
  - Handling of unknown words via suffix heuristics

References:
  Jurafsky & Martin, Speech and Language Processing, Ch. 8
  Anwar et al. (2007), HMM-based Urdu POS Tagger, ITJ.
"""
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .data_loader import DataLoader, Sentence

# Sentinel tags for sequence boundaries
START_TAG = "<START>"
END_TAG = "<END>"
UNK = "<UNK>"


class HMMTagger:
    """First-order Hidden Markov Model POS tagger.

    Training
    --------
    Call :meth:`train` with a list of sentences (each a list of (word, tag)
    tuples).  The model estimates:

    * **Transition probabilities** P(tag_i | tag_{i-1})
    * **Emission probabilities**   P(word | tag)

    Both are smoothed with add-k Laplace smoothing.

    Decoding
    --------
    Call :meth:`viterbi` (or :meth:`tag`) to decode a raw word sequence using
    the Viterbi dynamic-programming algorithm.
    """

    def __init__(self, smoothing_k: float = 1.0):
        """
        Parameters
        ----------
        smoothing_k:
            Laplace smoothing constant (default 1.0 = full Laplace smoothing).
            Smaller values (e.g., 0.01) give less smoothing for rare events.
        """
        self.smoothing_k = smoothing_k

        # Raw counts (populated by train())
        self._tag_counts: Dict[str, int] = {}
        self._bigram_counts: Dict[str, Dict[str, int]] = {}  # [prev][curr]
        self._emission_counts: Dict[str, Dict[str, int]] = {}  # [tag][word]

        # Derived probability tables (log-space)
        self.log_trans: Dict[str, Dict[str, float]] = {}   # log P(curr | prev)
        self.log_emit: Dict[str, Dict[str, float]] = {}    # log P(word | tag)

        # Tag vocabulary
        self.tags: List[str] = []
        self.vocab: set = set()
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sentences: List[Sentence]) -> None:
        """Estimate HMM parameters from annotated sentences.

        Parameters
        ----------
        sentences:
            List of sentences; each sentence is a list of (word, tag) pairs.
        """
        self._reset()
        for sentence in sentences:
            prev_tag = START_TAG
            self._increment_bigram(START_TAG, START_TAG)  # init counts
            for word, tag in sentence:
                self.vocab.add(word)
                self._increment_tag(tag)
                self._increment_bigram(prev_tag, tag)
                self._increment_emission(tag, word)
                prev_tag = tag
            self._increment_bigram(prev_tag, END_TAG)

        self.tags = [t for t in self._tag_counts if t not in (START_TAG, END_TAG)]
        self._compute_log_probabilities()
        self._trained = True

    # ------------------------------------------------------------------
    # Viterbi decoding
    # ------------------------------------------------------------------

    def viterbi(self, words: List[str]) -> Tuple[List[str], float]:
        """Run Viterbi algorithm on a word sequence.

        Parameters
        ----------
        words:
            List of (tokenised) words to tag.

        Returns
        -------
        best_tags:
            Most likely POS tag sequence.
        log_prob:
            Log-probability of the best path.
        """
        if not self._trained:
            raise RuntimeError("Model is not trained. Call train() first.")
        if not words:
            return [], 0.0

        n = len(words)
        T = len(self.tags)
        tag_index = {t: i for i, t in enumerate(self.tags)}

        # viterbi[t][i] = max log-prob of any path ending in tag i at position t
        NEG_INF = float('-inf')
        viterbi = [[NEG_INF] * T for _ in range(n)]
        backpointer = [[0] * T for _ in range(n)]

        # --- Initialisation (t = 0) ---
        word0 = words[0]
        for i, tag in enumerate(self.tags):
            log_p_trans = self._log_trans(START_TAG, tag)
            log_p_emit = self._log_emit(tag, word0)
            viterbi[0][i] = log_p_trans + log_p_emit
            backpointer[0][i] = 0  # no predecessor

        # --- Recursion ---
        for t in range(1, n):
            word = words[t]
            for i, curr_tag in enumerate(self.tags):
                log_p_emit = self._log_emit(curr_tag, word)
                best_prev = NEG_INF
                best_idx = 0
                for j, prev_tag in enumerate(self.tags):
                    score = viterbi[t - 1][j] + self._log_trans(prev_tag, curr_tag)
                    if score > best_prev:
                        best_prev = score
                        best_idx = j
                viterbi[t][i] = best_prev + log_p_emit
                backpointer[t][i] = best_idx

        # --- Termination ---
        best_final = NEG_INF
        best_final_idx = 0
        for i, tag in enumerate(self.tags):
            score = viterbi[n - 1][i] + self._log_trans(tag, END_TAG)
            if score > best_final:
                best_final = score
                best_final_idx = i

        # --- Backtrace ---
        best_tags = [None] * n
        best_tags[n - 1] = self.tags[best_final_idx]
        prev_idx = best_final_idx
        for t in range(n - 2, -1, -1):
            prev_idx = backpointer[t + 1][prev_idx]
            best_tags[t] = self.tags[prev_idx]

        return best_tags, best_final

    def tag(self, words: List[str]) -> List[str]:
        """Tag a list of words; convenience wrapper around :meth:`viterbi`."""
        tags, _ = self.viterbi(words)
        return tags

    def tag_sentence_str(self, sentence: str) -> List[Tuple[str, str]]:
        """Tag a whitespace-tokenised Urdu sentence string.

        Returns a list of (word, tag) pairs.
        """
        words = sentence.strip().split()
        tags = self.tag(words)
        return list(zip(words, tags))

    # ------------------------------------------------------------------
    # Probability tables (internal)
    # ------------------------------------------------------------------

    def _compute_log_probabilities(self) -> None:
        """Build log-probability look-up tables from raw counts."""
        V_words = len(self.vocab)
        V_tags = len(self._tag_counts)
        k = self.smoothing_k

        # --- Transition probabilities P(curr | prev) ---
        all_tags_with_start = [START_TAG] + self.tags
        for prev in all_tags_with_start:
            self.log_trans[prev] = {}
            prev_count = self._tag_counts.get(prev, 0)
            transitions = self._bigram_counts.get(prev, {})
            # denominator: count(prev) + k * |tags + END|
            denom = prev_count + k * (V_tags + 1)
            for curr in self.tags + [END_TAG]:
                num = transitions.get(curr, 0) + k
                self.log_trans[prev][curr] = math.log(num / denom)

        # --- Emission probabilities P(word | tag) ---
        for tag in self.tags:
            self.log_emit[tag] = {}
            tag_count = self._tag_counts.get(tag, 0)
            emissions = self._emission_counts.get(tag, {})
            denom = tag_count + k * V_words
            for word in self.vocab:
                num = emissions.get(word, 0) + k
                self.log_emit[tag][word] = math.log(num / denom)
            # UNK emission
            self.log_emit[tag][UNK] = math.log(k / denom)

    def _log_trans(self, prev: str, curr: str) -> float:
        """Look up log P(curr | prev)."""
        return self.log_trans.get(prev, {}).get(curr, float('-inf'))

    def _log_emit(self, tag: str, word: str) -> float:
        """Look up log P(word | tag), falling back to UNK."""
        tag_emit = self.log_emit.get(tag, {})
        if word in tag_emit:
            return tag_emit[word]
        # Unknown word handling: use UNK probability adjusted by suffix
        return tag_emit.get(UNK, float('-inf')) + self._suffix_bonus(tag, word)

    def _suffix_bonus(self, tag: str, word: str) -> float:
        """Heuristic log-probability bonus for unknown words based on Urdu morphology.

        Urdu verbs often end with specific suffixes; adjectives/nouns follow
        patterns.  Returns a small additive log-probability bonus.
        """
        # Common Urdu verb endings (past tense masculine singular)
        verb_past_endings = ('ا', 'یا', 'ئی', 'ئے')
        # Present/copula
        verb_pres_endings = ('ہے', 'ہیں', 'ہو', 'ہوں')
        # Adjective (masculine) ending in ا
        adj_endings = ('ا', 'ے', 'ی')

        bonus = 0.0
        if tag in ('VBD',) and word.endswith(verb_past_endings):
            bonus = 1.0
        elif tag in ('VBZ',) and word.endswith(verb_pres_endings):
            bonus = 1.0
        elif tag in ('JJ',) and word.endswith(adj_endings):
            bonus = 0.3
        elif tag == 'PUNC' and word in ('۔', '،', '؟', '!', ':'):
            bonus = 2.0
        return bonus

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._tag_counts = defaultdict(int)
        self._bigram_counts = defaultdict(lambda: defaultdict(int))
        self._emission_counts = defaultdict(lambda: defaultdict(int))
        self.log_trans = {}
        self.log_emit = {}
        self.tags = []
        self.vocab = set()
        self._trained = False

    def _increment_tag(self, tag: str) -> None:
        self._tag_counts[tag] += 1

    def _increment_bigram(self, prev: str, curr: str) -> None:
        self._bigram_counts[prev][curr] += 1
        if prev not in self._tag_counts:
            self._tag_counts[prev] = 0

    def _increment_emission(self, tag: str, word: str) -> None:
        self._emission_counts[tag][word] += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_top_emissions(self, tag: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return the n most probable words for a given tag."""
        if not self._trained:
            raise RuntimeError("Model not trained.")
        emissions = self._emission_counts.get(tag, {})
        total = self._tag_counts.get(tag, 1)
        sorted_items = sorted(emissions.items(), key=lambda x: -x[1])[:n]
        return [(word, count / total) for word, count in sorted_items]

    def get_top_transitions(self, prev_tag: str, n: int = 10) -> List[Tuple[str, float]]:
        """Return the n most probable next tags given a previous tag."""
        if not self._trained:
            raise RuntimeError("Model not trained.")
        trans = self._bigram_counts.get(prev_tag, {})
        total = sum(trans.values()) or 1
        sorted_items = sorted(trans.items(), key=lambda x: -x[1])[:n]
        return [(tag, count / total) for tag, count in sorted_items if tag != END_TAG]
