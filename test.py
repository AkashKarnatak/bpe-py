import os
from bpe import BasicTokenizer
from regex_bpe import RegexTokenizer
from utils import load_bpe

with open("./taylorswift.txt", "r") as f:
    text = f.read()

if not os.path.exists("./basic-merges.bpe"):
    # train tokenizer
    print("Training basic tokenizer")
    enc = BasicTokenizer()
    enc.train(text, 1000, verbose=True, bpe_path="basic-merges.bpe")

basic_ranks = load_bpe("./basic-merges.bpe")
enc = BasicTokenizer(basic_ranks)
inp = "Listen to me, mister. You're my knight in shining armor. Don't you forget it. You're going to get back on that horse, and I'm going to be right behind you, holding on tight, and away we're gonna , go, go!"
ids = enc.encode(inp)
print(ids)
out = enc.decode(ids)
assert inp == out
print("Basic tokenizer is working properly!\n")

if not os.path.exists("./regex-merges.bpe"):
    # train tokenizer
    print("Training regex tokenizer")
    enc = RegexTokenizer()
    enc.train(text, 1000, verbose=True, bpe_path="regex-merges.bpe")

regex_ranks = load_bpe("./regex-merges.bpe")
enc = RegexTokenizer(regex_ranks)
inp = "Listen to me, mister. You're my knight in shining armor. Don't you forget it. You're going to get back on that horse, and I'm going to be right behind you, holding on tight, and away we're gonna , go, go!"
ids = enc.encode(inp)
print(ids)
out = enc.decode(ids)
assert inp == out
print("Regex tokenizer is working properly!\n")

print("Loading GPT-4 merges file...")
gpt4_ranks = load_bpe("./o200k_base.tiktoken")
enc = RegexTokenizer(gpt4_ranks)
inp = "hello world!!!? (안녕하세요!) lol123 😉"
ids = enc.encode(inp)
print(ids)
out = enc.decode(ids)
print(out)
assert inp == out
print("🎉 Woohoo! Was able to produce identical results as GPT-4 tokenizer")
