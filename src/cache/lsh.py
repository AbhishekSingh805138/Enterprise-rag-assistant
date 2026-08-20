"""Locality-sensitive hashing, so the semantic cache stops scanning.

Every cache lookup loaded every entry for the scope — each with its full
embedding — and computed cosine similarity in Python. Fine at a hundred
entries; at fifty thousand it is tens of megabytes of JSON parsed and
1536 multiplications each, on the request path, to answer a question the
cache exists to make *faster*.

Random-hyperplane LSH gives the missing index. Each vector is projected
onto a fixed set of random hyperplanes; the sign of each projection is
one bit, and vectors pointing in similar directions agree on most bits.
Grouping the bits into bands and matching on *any* band retrieves near
neighbours with a cheap equality lookup that an ordinary B-tree index can
serve.

Two properties that matter for using it here:

* **It prunes; it does not decide.** Candidates still go through exact
  cosine similarity against the threshold, so a false positive from the
  hash costs one comparison and can never return a wrong answer.
* **A false negative is a cache miss.** The system recomputes the answer,
  which is correct, just not free. That is the right direction to fail in
  for a cache, and it is why the band count is tuned for recall rather
  than for the smallest possible candidate set.

The hyperplanes are generated from a fixed seed so every replica derives
the same buckets — a per-process random basis would make one replica's
cache entries invisible to the next.
"""
from __future__ import annotations

import logging
import math
import random

logger = logging.getLogger(__name__)

# 4 bands x 8 bits. At the cache's default 0.95 threshold two vectors
# disagree on any given bit with probability arccos(0.95)/pi ~= 0.10, so
# a single 8-bit band matches with probability ~0.43 and at least one of
# four bands with ~0.90. Fewer, wider bands would prune harder and miss
# more; more, narrower bands would retrieve most of the table again.
BANDS = 4
BITS_PER_BAND = 8
SEED = 0x5EED  # fixed so all replicas bucket identically

_planes_cache: dict[int, list[list[float]]] = {}


def _planes(dimension: int) -> list[list[float]]:
    """Random hyperplanes for *dimension*, cached per process."""
    planes = _planes_cache.get(dimension)
    if planes is not None:
        return planes

    rng = random.Random(SEED)
    planes = [
        [rng.gauss(0.0, 1.0) for _ in range(dimension)]
        for _ in range(BANDS * BITS_PER_BAND)
    ]
    _planes_cache[dimension] = planes
    return planes


def signature(vector: list[float]) -> list[int]:
    """Band hashes for *vector*: one integer per band.

    Returns an empty list for an unusable vector (empty, or all zeros),
    which callers treat as "not indexable" rather than bucketing every
    such vector together.
    """
    if not vector:
        return []
    if not any(vector):
        return []

    planes = _planes(len(vector))
    bits: list[int] = []
    for plane in planes:
        projection = sum(v * p for v, p in zip(vector, plane, strict=True))
        bits.append(1 if projection >= 0 else 0)

    bands: list[int] = []
    for band in range(BANDS):
        start = band * BITS_PER_BAND
        value = 0
        for bit in bits[start : start + BITS_PER_BAND]:
            value = (value << 1) | bit
        # Distinguish band 0's hash from band 1's, so a query cannot
        # match a stored entry on the wrong band's column value.
        bands.append(value | (band << BITS_PER_BAND))
    return bands


def expected_recall(threshold: float) -> float:
    """Probability that a true neighbour at *threshold* shares a band.

    Used to document and test the parameter choice rather than leaving
    BANDS and BITS_PER_BAND as magic numbers nobody can justify.
    """
    threshold = max(-1.0, min(1.0, threshold))
    bit_agreement = 1.0 - math.acos(threshold) / math.pi
    band_match = bit_agreement**BITS_PER_BAND
    return 1.0 - (1.0 - band_match) ** BANDS
