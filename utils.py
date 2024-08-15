import base64

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


GPT4_SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endofprompt|>",
]


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
