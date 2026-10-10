"""
perceptron_tagger.py - Averaged perceptron POS tagger (stdlib only).

Greedy left-to-right decoding with Collins (2002) weight averaging.

Public interface:
    tagger = AveragedPerceptron()
    tagger.train(tagged_sents, n_epochs=5, seed=42, lang_filter=None)
    tags   = tagger.predict(sent)         # sent: [(form, lang), ...]
    mb     = tagger.save(path)
    tagger = AveragedPerceptron.load(path)

A tagged sentence is a list of (form, lang, gold_tag) tuples.
An untagged sentence is a list of (form, lang) tuples.
lang_filter: if set, only update weights for tokens whose lang == lang_filter,
             but full sentence context is always used for features.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

START = "__START__"
BOS   = "__BOS__"
EOS   = "__EOS__"


# ---------------------------------------------------------------------------
# Script/shape utilities
# ---------------------------------------------------------------------------

def _has_hangul(s: str) -> bool:
    return any(
        '가' <= c <= '힣' or 'ᄀ' <= c <= 'ᇿ' or '㄰' <= c <= '㆏'
        for c in s
    )


def _has_latin(s: str) -> bool:
    return any(c.isascii() and c.isalpha() for c in s)


def _shape(form: str) -> str:
    """
    Compressed shape string; consecutive identical codes are collapsed.
    X=upper-ASCII  x=lower-ASCII  d=digit  H=Hangul  p=other
    """
    out: list[str] = []
    prev: str | None = None
    for c in form:
        if c.isdigit():
            code = 'd'
        elif c.isascii() and c.isupper():
            code = 'X'
        elif c.isascii() and c.islower():
            code = 'x'
        elif '가' <= c <= '힣' or 'ᄀ' <= c <= 'ᇿ' or '㄰' <= c <= '㆏':
            code = 'H'
        else:
            code = 'p'
        if code != prev:
            out.append(code)
            prev = code
    return ''.join(out) or 'p'


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def get_features(
    form: str, lang: str,
    p1_form: str, n1_form: str,
    p1_lang: str, n1_lang: str,
    p1_tag:  str, p2_tag:  str,
) -> list[str]:
    """Return the feature string list for one token in its sentence context."""
    lo = form.lower()
    return [
        "bias",
        f"w={lo}",
        f"lang={lang}",
        f"shape={_shape(form)}",
        f"hgl={_has_hangul(form)}",
        f"lat={_has_latin(form)}",
        f"dig={form.isdigit()}",
        # character n-gram prefixes / suffixes (up to 3)
        f"pre1={lo[:1]}",
        f"pre2={lo[:2]}",
        f"pre3={lo[:3]}",
        f"suf1={lo[-1:]}",
        f"suf2={lo[-2:]}",
        f"suf3={lo[-3:]}",
        # surrounding token forms
        f"p1={p1_form}",
        f"n1={n1_form}",
        # surrounding language labels
        f"p1.lang={p1_lang}",
        f"n1.lang={n1_lang}",
        # previous predicted tags (Markov context)
        f"p1.tag={p1_tag}",
        f"p2.tag={p2_tag}",
        # joint features
        f"p1.tag+lang={p1_tag}+{lang}",
    ]


# ---------------------------------------------------------------------------
# Averaged perceptron
# ---------------------------------------------------------------------------

class AveragedPerceptron:
    """
    Averaged perceptron POS tagger.

    Weight averaging follows Collins (2002): the average is taken over all
    token-processing steps (not just mistake steps), so the denominator equals
    total tokens processed across all epochs.
    """

    def __init__(self) -> None:
        self.tags:    list[str]               = []
        self.weights: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Scoring / decoding
    # ------------------------------------------------------------------

    def _score(self, feats: list[str]) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        for f in feats:
            wf = self.weights.get(f)
            if wf:
                for tag, w in wf.items():
                    scores[tag] += w
        return scores

    def _best(self, feats: list[str]) -> str:
        scores = self._score(feats)
        if not scores:
            return self.tags[0] if self.tags else "NOUN"
        return max(self.tags, key=lambda t: scores.get(t, 0.0))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        tagged_sents: list[list[tuple]],
        n_epochs:    int = 5,
        seed:        int = 42,
        lang_filter: str | None = None,
    ) -> None:
        """
        Train on a list of tagged sentences.

        Each sentence is a list of (form, lang, gold_tag) tuples.
        If lang_filter is set, weight updates only apply to tokens where
        lang == lang_filter; all other tokens still contribute context.
        """
        # Collect tag vocabulary from training data
        all_tags: set[str] = set()
        for sent in tagged_sents:
            for _, _, t in sent:
                all_tags.add(t)
        self.tags = sorted(all_tags)

        # Training-time weight storage
        W: defaultdict = defaultdict(lambda: defaultdict(float))   # current weights
        T: defaultdict = defaultdict(lambda: defaultdict(float))   # accumulated totals
        S: defaultdict = defaultdict(lambda: defaultdict(int))     # last-update stamps
        self.weights = W   # _best/_score read from self.weights during training

        step = 0  # total tokens processed (denominator for averaging)
        rng  = random.Random(seed)
        idx_order = list(range(len(tagged_sents)))

        for ep in range(n_epochs):
            rng.shuffle(idx_order)
            corr = filt_tot = 0

            for idx in idx_order:
                sent  = tagged_sents[idx]
                n     = len(sent)
                forms = [t[0] for t in sent]
                langs = [t[1] for t in sent]
                golds = [t[2] for t in sent]
                p1_tag = p2_tag = START

                for j in range(n):
                    p1f  = forms[j-1].lower() if j > 0    else BOS
                    n1f  = forms[j+1].lower() if j < n-1  else EOS
                    p1l  = langs[j-1]          if j > 0    else BOS
                    n1l  = langs[j+1]          if j < n-1  else EOS

                    feats = get_features(
                        forms[j], langs[j], p1f, n1f, p1l, n1l, p1_tag, p2_tag
                    )
                    guess = self._best(feats)
                    gold  = golds[j]
                    step += 1

                    do_update = (lang_filter is None or langs[j] == lang_filter)
                    if do_update:
                        filt_tot += 1
                        if guess == gold:
                            corr += 1
                        else:
                            # Perceptron update: +1 for gold, -1 for guess
                            for f in feats:
                                # gold
                                n_steps = step - S[f][gold]
                                T[f][gold] += n_steps * W[f][gold]
                                S[f][gold]  = step
                                W[f][gold] += 1.0
                                # guess
                                n_steps = step - S[f][guess]
                                T[f][guess] += n_steps * W[f][guess]
                                S[f][guess]  = step
                                W[f][guess] -= 1.0

                    p2_tag, p1_tag = p1_tag, guess

            pct = 100.0 * corr / filt_tot if filt_tot else 0.0
            tag_lbl = f" [lang={lang_filter}]" if lang_filter else ""
            print(f"  Epoch {ep+1}/{n_epochs}: acc={pct:.2f}%"
                  f"  ({corr:,}/{filt_tot:,} filtered tokens{tag_lbl})")

        # ------------------------------------------------------------------
        # Compute averaged weights (Collins 2002)
        # ------------------------------------------------------------------
        n_total = step or 1
        avg: dict[str, dict[str, float]] = {}
        for feat, tw in W.items():
            tg: dict[str, float] = {}
            for tag, w in tw.items():
                total = T[feat][tag] + (n_total - S[feat][tag]) * w
                a     = total / n_total
                if abs(a) > 1e-9:
                    tg[tag] = round(a, 9)
            if tg:
                avg[feat] = tg

        self.weights = avg   # replace live weights with averaged weights
        print(f"  Averaged over {n_total:,} steps -> {len(self.weights):,} non-zero features")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, sent: list[tuple]) -> list[str]:
        """
        Predict UPOS tags for a sentence.
        Each element: (form, lang) or (form, lang, gold_tag) — gold ignored.
        Returns a list of predicted tags (same length as sent).
        """
        n     = len(sent)
        forms = [t[0] for t in sent]
        langs = [t[1] for t in sent]
        p1_tag = p2_tag = START
        preds: list[str] = []

        for j in range(n):
            p1f = forms[j-1].lower() if j > 0   else BOS
            n1f = forms[j+1].lower() if j < n-1 else EOS
            p1l = langs[j-1]          if j > 0   else BOS
            n1l = langs[j+1]          if j < n-1 else EOS

            feats = get_features(
                forms[j], langs[j], p1f, n1f, p1l, n1l, p1_tag, p2_tag
            )
            guess = self._best(feats)
            preds.append(guess)
            p2_tag, p1_tag = p1_tag, guess

        return preds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> int:
        """Save model to JSON. Returns file size in MB."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type":    "AveragedPerceptron",
            "tags":    self.tags,
            "weights": self.weights,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        mb = path.stat().st_size // (1024 * 1024)
        print(f"  Saved {path.name}: {len(self.weights):,} features, {mb} MB")
        return mb

    @classmethod
    def load(cls, path: Path) -> "AveragedPerceptron":
        """Load a model from JSON."""
        with path.open(encoding="utf-8") as f:
            d = json.load(f)
        m = cls()
        m.tags    = d["tags"]
        m.weights = d["weights"]
        return m
