import base64
import regex as re
import itertools as it
from collections import Counter
from utils import GPT4_SPECIAL_TOKENS, GPT4_SPLIT_PATTERN


class RegexTokenizer:
    def __init__(self, ranks: dict[bytes, int] = {}, pat_str: str = GPT4_SPLIT_PATTERN):
        self.ranks = ranks
        self.pat = re.compile(pat_str)

    def merge(
        self, words: list[list[bytes]], pair: tuple[bytes, bytes], token: bytes
    ) -> list[list[bytes]]:
        new_words: list[list[bytes]] = []
        for ids in words:
            new_ids: list[bytes] = []
            i = 0

            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    new_ids.append(token)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1

            new_words.append(new_ids)
        return new_words

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
        bpe_path: str = "merges.bpe",
    ):
        assert (
            vocab_size > 256
        ), "We need atleast 256 elements to represent all the bytes!"

        words = [
            [bytes([x]) for x in word.encode("utf-8")]
            for word in self.pat.findall(text)
        ]

        ranks: dict[bytes, int] = {}

        with open(bpe_path, "wb") as f:
            for i in range(2**8):
                ranks[bytes([i])] = i
                f.write(
                    base64.b64encode(bytes([i])) + b" " + str(i).encode("utf-8") + b"\n"
                )

            while len(ranks) < vocab_size:
                # find freq of all pairs
                counter = Counter(
                    it.chain.from_iterable([zip(ids, ids[1:]) for ids in words])
                )

                if len(counter) == 0:
                    raise Exception("Cannot compress further. Try reducing vocab_size")

                top_pair = counter.most_common(1)[0][0]
                top_pair_token = b"".join(top_pair)

                if verbose:
                    print(f"Merging {top_pair} into {top_pair_token}")

                ranks[top_pair_token] = len(ranks)
                f.write(
                    base64.b64encode(top_pair_token)
                    + b" "
                    + str(len(ranks)).encode("utf-8")
                    + b"\n"
                )
                words = self.merge(words, top_pair, top_pair_token)

    def encode(self, text: str) -> list[list[bytes]]:
        special_token_pat = (
            "(" + "|".join([re.escape(x) for x in GPT4_SPECIAL_TOKENS]) + ")"
        )
        words = re.split(special_token_pat, text)
        new_words: list[list[bytes]] = []
        for word in words:
            if word in GPT4_SPECIAL_TOKENS:
                new_words.append([word.encode("utf-8")])
            else:
                new_words.extend(
                    [
                        [bytes([x]) for x in word.encode("utf-8")]
                        for word in self.pat.findall(word)
                    ]
                )
        words = new_words

        # compress
        new_words: list[list[bytes]] = []
        for ids in words:
            while True:
                if len(ids) == 1:
                    break

                pairs = zip(ids, ids[1:])
                pair = min(
                    pairs, key=lambda x: self.ranks.get(b"".join(x), float("inf"))
                )
                pair_token = b"".join(pair)
                if pair_token not in self.ranks:
                    break
                ids = self.merge([ids], pair, pair_token)[0]
            new_words.append(ids)
        return new_words

    def decode(self, words: list[list[bytes]]) -> str:
        return b"".join([b"".join(ids) for ids in words]).decode("utf-8")
