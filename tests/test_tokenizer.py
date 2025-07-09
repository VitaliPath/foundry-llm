from pathlib import Path

from foundry_llm.model.tokenizer import Tokenizer

CORPUS = Path(__file__).parent / "tiny_shakespeare_sample.txt"


def test_round_trip():
    tk = Tokenizer(CORPUS)
    tk.train()
    original = "To be, or not to be."
    ids = tk.encode(original)
    assert tk.decode(ids) == original
    assert tk.vocab_size > 0
