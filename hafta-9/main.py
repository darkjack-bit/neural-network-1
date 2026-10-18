
from collections import Counter
from pathlib import Path

import regex
import tiktoken


GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""


def get_stats(ids):
    return Counter(zip(ids, ids[1:]))


def merge(ids, pair, idx):
    result = []
    i = 0

    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(idx)
            i += 2
        else:
            result.append(ids[i])
            i += 1

    return result


class BPETokenizer:
    def __init__(self, pattern=None):
        self.pattern = regex.compile(pattern) if pattern else None
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.history = []

    def split(self, text):
        if not self.pattern:
            return [text]
        return self.pattern.findall(text)

    def train(self, text, vocab_size):
        assert vocab_size >= 256

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.history = []

        chunks = [
            list(piece.encode("utf-8"))
            for piece in self.split(text)
        ]

        for idx in range(256, vocab_size):
            stats = Counter()

            for ids in chunks:
                stats.update(get_stats(ids))

            if not stats:
                break

            pair = max(stats, key=stats.get)

            chunks = [
                merge(ids, pair, idx)
                for ids in chunks
            ]

            self.merges[pair] = idx
            self.vocab[idx] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )

            self.history.append(
                (pair, idx, stats[pair])
            )

    def encode_chunk(self, text):
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)

            pair = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf"))
            )

            if pair not in self.merges:
                break

            ids = merge(ids, pair, self.merges[pair])

        return ids

    def encode(self, text):
        ids = []

        for piece in self.split(text):
            ids.extend(self.encode_chunk(piece))

        return ids

    def decode(self, ids):
        data = b"".join(self.vocab[idx] for idx in ids)
        return data.decode("utf-8", errors="replace")


def show_merges(tokenizer):
    for i, (pair, idx, count) in enumerate(tokenizer.history[:20], 1):
        left = tokenizer.vocab[pair[0]]
        right = tokenizer.vocab[pair[1]]
        combined = tokenizer.vocab[idx]

        print(
            i,
            left.decode("utf-8", errors="replace"),
            "+",
            right.decode("utf-8", errors="replace"),
            "->",
            combined.decode("utf-8", errors="replace"),
            "count:",
            count
        )


def compression(tokenizer, text):
    byte_count = len(text.encode("utf-8"))
    token_count = len(tokenizer.encode(text))
    return byte_count / token_count


def compare_tiktoken(tr, en, own_tokenizer):
    enc = tiktoken.get_encoding("cl100k_base")

    for name, text in [
        ("Türkçe", tr),
        ("İngilizce", en)
    ]:
        tokens = enc.encode(text)
        words = len(text.split())

        print(name)
        print("Kelime:", words)
        print("Token:", len(tokens))
        print("Token/Kelime:", len(tokens) / words)

    own_tokens = own_tokenizer.encode(tr)

    print("Kendi Türkçe tokenizer")
    print("Token:", len(own_tokens))
    print("Token/Kelime:", len(own_tokens) / len(tr.split()))


def main():
    path = Path("input.txt")
    text = path.read_text(encoding="utf-8")

    if not text:
        raise ValueError("Eğitim dosyası boş olamaz.")

    print("EĞİTİM VERİSİ")
    print("Dosya:", path)
    print("Karakter:", len(text))
    print("Byte:", len(text.encode("utf-8")))

    sentence = "Çığ, ışık, öykü, şüphe, güzel 😊"

    print("\nUNICODE VE UTF-8")

    for char in sentence:
        print(
            repr(char),
            f"U+{ord(char):04X}",
            list(char.encode("utf-8")),
            len(char.encode("utf-8"))
        )

    basic = BPETokenizer()
    basic.train(text, 400)

    print("\nTEMEL BPE")
    print("Vocab boyutu:", len(basic.vocab))
    show_merges(basic)
    print("Sıkıştırma:", compression(basic, text))

    example = "İstanbul'da çiçekler açtı 😊🚀"

    ids = basic.encode(example)
    decoded = basic.decode(ids)

    print("\nENCODE VE DECODE")
    print("Orijinal:", example)
    print("Tokenlar:", ids)
    print("Decode:", decoded)
    print("Eşit mi:", decoded == example)
    print("Geçersiz UTF-8:", basic.decode([196]))

    advanced = BPETokenizer(GPT4_PATTERN)
    advanced.train(text, 400)

    print("\nREGEX BPE")
    print("Vocab boyutu:", len(advanced.vocab))
    show_merges(advanced)
    print("Sıkıştırma:", compression(advanced, text))

    print("\nREGEX PARÇALAMA")
    print(advanced.split("Ankara'dan"))
    print(advanced.split("İstanbul'da"))

    print("\nMERGE KARŞILAŞTIRMASI")

    for i in range(min(20, len(basic.history), len(advanced.history))):
        a = basic.vocab[basic.history[i][1]]
        b = advanced.vocab[advanced.history[i][1]]
        print(i + 1, repr(a), repr(b))

    advanced_ids = advanced.encode(example)
    advanced_decoded = advanced.decode(advanced_ids)

    print("\nREGEX ENCODE VE DECODE")
    print("Tokenlar:", advanced_ids)
    print("Decode:", advanced_decoded)
    print("Eşit mi:", advanced_decoded == example)

    tr = (
        "Yapay zekâ sistemleri büyük miktarda metinden öğrenir. "
        "Türkçede kelimelere ekler getirildiğinde farklı biçimler oluşur."
    )

    en = (
        "Artificial intelligence systems learn from large amounts of text. "
        "In Turkish, adding suffixes to words creates different forms."
    )

    print("\nTIKTOKEN KARŞILAŞTIRMASI")
    compare_tiktoken(tr, en, advanced)


if __name__ == "__main__":
    main()
