"""
Urdu HMM POS Tagger — Evaluation Module
Computes accuracy, per-tag precision/recall/F1, and confusion matrix.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_loader import Sentence
from .hmm_tagger import HMMTagger


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    tagger: HMMTagger,
    test_sentences: List[Sentence],
) -> Dict:
    """Run the tagger on test_sentences and return evaluation metrics.

    Returns
    -------
    dict with keys:
        accuracy          — overall token-level accuracy
        per_tag           — dict[tag] → {precision, recall, f1, support}
        confusion_matrix  — np.ndarray, shape (n_tags, n_tags)
        tag_list          — ordered list of tags (axis labels for confusion_matrix)
        total_tokens      — number of evaluated tokens
        correct_tokens    — number of correctly tagged tokens
        oov_accuracy      — accuracy on out-of-vocabulary (unseen) words
        iv_accuracy       — accuracy on in-vocabulary words
    """
    all_gold: List[str] = []
    all_pred: List[str] = []
    oov_correct = oov_total = 0
    iv_correct = iv_total = 0

    for sentence in test_sentences:
        words = [w for w, _ in sentence]
        gold_tags = [t for _, t in sentence]
        pred_tags = tagger.tag(words)

        for word, gold, pred in zip(words, gold_tags, pred_tags):
            all_gold.append(gold)
            all_pred.append(pred)
            is_oov = word not in tagger.vocab
            if is_oov:
                oov_total += 1
                if gold == pred:
                    oov_correct += 1
            else:
                iv_total += 1
                if gold == pred:
                    iv_correct += 1

    # Overall accuracy
    correct = sum(g == p for g, p in zip(all_gold, all_pred))
    total = len(all_gold)
    accuracy = correct / total if total else 0.0

    # Per-tag metrics
    tag_list = sorted(set(all_gold) | set(all_pred))
    per_tag = _per_tag_metrics(all_gold, all_pred, tag_list)

    # Confusion matrix
    cm = _confusion_matrix(all_gold, all_pred, tag_list)

    return {
        'accuracy': accuracy,
        'per_tag': per_tag,
        'confusion_matrix': cm,
        'tag_list': tag_list,
        'total_tokens': total,
        'correct_tokens': correct,
        'oov_accuracy': oov_correct / oov_total if oov_total else None,
        'iv_accuracy': iv_correct / iv_total if iv_total else None,
        'oov_total': oov_total,
        'iv_total': iv_total,
    }


def print_report(results: Dict, top_n_errors: int = 10) -> None:
    """Pretty-print evaluation results to stdout."""
    acc = results['accuracy']
    total = results['total_tokens']
    correct = results['correct_tokens']

    print("=" * 60)
    print("EVALUATION REPORT — Urdu HMM POS Tagger")
    print("=" * 60)
    print(f"Overall Accuracy : {acc:.4f}  ({correct}/{total} tokens correct)")
    oov_acc = results.get('oov_accuracy')
    iv_acc = results.get('iv_accuracy')
    if iv_acc is not None:
        print(f"In-Vocab Accuracy: {iv_acc:.4f}  ({results['iv_total']} tokens)")
    if oov_acc is not None:
        print(f"OOV Accuracy     : {oov_acc:.4f}  ({results['oov_total']} tokens)")
    print()

    # Per-tag table
    print(f"{'TAG':<8} {'PREC':>6} {'REC':>6} {'F1':>6} {'SUPPORT':>8}")
    print("-" * 42)
    per_tag = results['per_tag']
    for tag in sorted(per_tag, key=lambda t: -per_tag[t]['support']):
        m = per_tag[tag]
        print(
            f"{tag:<8} {m['precision']:>6.3f} {m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} {m['support']:>8}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _per_tag_metrics(
    gold: List[str],
    pred: List[str],
    tag_list: List[str],
) -> Dict[str, Dict]:
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    metrics = {}
    for tag in tag_list:
        t = tp[tag]
        f_p = fp[tag]
        f_n = fn[tag]
        support = t + f_n
        prec = t / (t + f_p) if (t + f_p) else 0.0
        rec = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[tag] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': support}
    return metrics


def _confusion_matrix(
    gold: List[str],
    pred: List[str],
    tag_list: List[str],
) -> np.ndarray:
    idx = {t: i for i, t in enumerate(tag_list)}
    n = len(tag_list)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            cm[idx[g]][idx[p]] += 1
    return cm


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    tagger: HMMTagger,
    test_sentences: List[Sentence],
    n: int = 20,
) -> List[Tuple[str, str, str, int]]:
    """Return the n most common tagging errors as (word, gold_tag, pred_tag, count) tuples."""
    error_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for sentence in test_sentences:
        words = [w for w, _ in sentence]
        gold_tags = [t for _, t in sentence]
        pred_tags = tagger.tag(words)
        for word, gold, pred in zip(words, gold_tags, pred_tags):
            if gold != pred:
                error_counts[(word, gold, pred)] += 1

    sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])
    return [(word, gold, pred, count) for (word, gold, pred), count in sorted_errors[:n]]
