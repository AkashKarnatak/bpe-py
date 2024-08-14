import base64
from collections import Counter


def load_bpe(bpe_path: str) -> dict[bytes, int]:
    ranks: dict[bytes, int] = {}
    with open(bpe_path, "rb") as f:
        for line in f.readlines():
            token_pair, rank = line.split()
            ranks[base64.b64decode(token_pair)] = int(rank)
    return ranks


def dump_bpe(ranks: dict[bytes, int]):
    with open("merges.bpe", "wb") as f:
        for pair, rank in ranks.items():
            f.writelines(base64.b64encode(pair) + b" " + str(rank).encode("utf-8"))


class BasicTokenizer:
    def __init__(self, ranks: dict[bytes, int] = {}):
        self.ranks = ranks

    def merge(
        self, ids: list[bytes], pair: tuple[bytes, bytes], token: bytes
    ) -> list[bytes]:
        new_ids: list[bytes] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(token)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

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

        ids = [bytes([x]) for x in text.encode("utf-8")]

        ranks: dict[bytes, int] = {}

        with open(bpe_path, "wb") as f:
            for i in range(2**8):
                ranks[bytes([i])] = i
                f.write(
                    base64.b64encode(bytes([i])) + b" " + str(i).encode("utf-8") + b"\n"
                )

            while len(ranks) < vocab_size:
                # find freq of all pairs
                counter = Counter(zip(ids, ids[1:]))

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
                ids = self.merge(ids, top_pair, top_pair_token)

    def encode(self, text: str) -> list[bytes]:
        ids = [bytes([x]) for x in text.encode("utf-8")]

        # compress
        while True:
            pairs = zip(ids, ids[1:])
            pair = min(pairs, key=lambda x: self.ranks.get(b"".join(x), float("inf")))
            pair_token = b"".join(pair)
            if pair_token not in self.ranks:
                break
            ids = self.merge(ids, pair, pair_token)
        return ids

    def decode(self, ids: list[bytes]) -> str:
        return b"".join(ids).decode("utf-8")
