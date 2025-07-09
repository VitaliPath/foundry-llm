from __future__ import annotations

from pathlib import Path
from typing import Dict, List


class Tokenizer:
    """
    A minimal character-level tokenizer.

    * `vocab`      – Ordered mapping char → int
    * `inverse`    – Reverse table int → char
    """

    def __init__(self, corpus_path: Path | str):
        self.corpus_path = Path(corpus_path)
        self.vocab: Dict[str, int] = {}
        self.inverse: List[str] = []

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def train(self) -> None:
        """Scan corpus and build char → id tables."""
        text = self.corpus_path.read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inverse = chars  # same order as enumerate

    def encode(self, text: str) -> List[int]:
        return [self.vocab[c] for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.inverse[i] for i in ids)

    # convenience
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
