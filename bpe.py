"""Educational character-level BPE, implemented from scratch.

Words are separated on whitespace and terminated by a special end-of-word
symbol. This is intentionally simpler than byte-level BPE used in GPT-4.
"""

from collections import Counter
import json
from pathlib import Path


class BPE:
    END_OF_WORD = "</w>"

    def __init__(self, num_merges: int = 100, min_frequency: int = 2):
        if num_merges < 0:
            raise ValueError("num_merges must be >= 0")
        if min_frequency < 1:
            raise ValueError("min_frequency must be >= 1")
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.merges: list[tuple[str, str]] = []
        self.vocab: set[str] = set()
        self.trained = False

    @staticmethod
    def _merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
        """Replace every non-overlapping occurrence of an adjacent pair."""
        merged = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return tuple(merged)

    def fit(self, corpus: str) -> "BPE":
        """Learn merge rules from a whitespace-separated text corpus."""
        word_counts = Counter(corpus.split())
        if not word_counts:
            raise ValueError("The training corpus is empty")

        word_symbols = Counter({
            tuple(list(word) + [self.END_OF_WORD]): frequency
            for word, frequency in word_counts.items()
        })
        self.merges = []
        self.vocab = {self.END_OF_WORD}
        self.vocab.update(char for word in word_counts for char in word)
        self.trained = False

        for _ in range(self.num_merges):
            pair_counts = Counter()
            for symbols, word_frequency in word_symbols.items():
                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += word_frequency

            if not pair_counts:
                break
            best_pair, frequency = pair_counts.most_common(1)[0]
            if frequency < self.min_frequency:
                break

            word_symbols = Counter({
                self._merge_pair(symbols, best_pair): word_frequency
                for symbols, word_frequency in word_symbols.items()
            })
            self.merges.append(best_pair)
            self.vocab.add("".join(best_pair))

        self.trained = True
        return self

    def encode(self, text: str) -> list[str]:
        """Tokenize text using the learned merge order.

        End-of-word markers are hidden in the returned tokens.
        Whitespace information is not retained in this basic implementation.
        """
        if not self.trained:
            raise RuntimeError("Call fit() before encode()")

        tokens = []
        for word in text.split():
            symbols = tuple(list(word) + [self.END_OF_WORD])
            for pair in self.merges:
                symbols = self._merge_pair(symbols, pair)
            for symbol in symbols:
                token = symbol.replace(self.END_OF_WORD, "")
                if token:
                    tokens.append(token)
        return tokens

    def save_vocab(self, path: str | Path) -> None:
        """Save the trained vocabulary and ordered merge rules as JSON."""
        if not self.trained:
            raise RuntimeError("Call fit() before save_vocab()")
        payload = {
            "tokens": sorted(self.vocab),
            "merges": [list(pair) for pair in self.merges],
            "num_merges_learned": len(self.merges),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_vocab(cls, path: str | Path) -> "BPE":
        """Restore a saved model without retraining; preserve merge order."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data.get("merges"), list) or not isinstance(data.get("tokens"), list):
            raise ValueError("Invalid vocabulary JSON: missing merges or tokens")
        model = cls(num_merges=len(data["merges"]))
        model.merges = [tuple(pair) for pair in data["merges"]]
        if any(len(pair) != 2 for pair in model.merges):
            raise ValueError("Invalid vocabulary: each merge must be a pair")
        model.vocab = set(data["tokens"])
        model.trained = True
        return model


def main():
    """Train on supplied corpus files or encode text with a saved vocabulary.

    Examples:
        python bpe.py
        python bpe.py train english.txt persian.txt
        python bpe.py encode results/vocab_fa.json "سلام دنیا"
    """
    import argparse
    import unicodedata
    from time import perf_counter

    parser = argparse.ArgumentParser(description="Character-level BPE from scratch")
    sub = parser.add_subparsers(dest="command")
    trainer = sub.add_parser("train", help="Train and save separate EN/FA models")
    trainer.add_argument("english", type=Path)
    trainer.add_argument("persian", type=Path)
    trainer.add_argument("--num-merges", type=int, default=200)
    trainer.add_argument("--min-frequency", type=int, default=2)
    trainer.add_argument("--output-dir", type=Path, default=Path("results"))
    encoder = sub.add_parser("encode", help="Tokenize using saved vocabulary")
    encoder.add_argument("vocab", type=Path)
    encoder.add_argument("text")
    args = parser.parse_args()

    if args.command == "train":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for language, path in (("en", args.english), ("fa", args.persian)):
            text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
            model = BPE(args.num_merges, args.min_frequency)
            start = perf_counter()
            model.fit(text)
            seconds = perf_counter() - start
            model.save_vocab(args.output_dir / f"vocab_{language}.json")
            print(f"{language}: merges={len(model.merges)}, vocab={len(model.vocab)}, "
                  f"time={seconds:.6f} seconds")
    elif args.command == "encode":
        print(BPE.from_vocab(args.vocab).encode(args.text))
    else:
        model = BPE(num_merges=3).fit("cat cat car")
        print("Merges:", model.merges)
        print("Tokens:", model.encode("cat car"))


if __name__ == "__main__":
    main()
