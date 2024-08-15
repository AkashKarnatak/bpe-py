# Byte Pair Encoding (BPE) Tokenizer

This repository provides an implementation of the Byte Pair Encoding (BPE) algorithm for text tokenization. The project includes two main Python scripts: `bpe.py` and `regex.py`, each offering different approaches for text tokenization using BPE.

## Overview

Byte Pair Encoding (BPE) is a data compression technique adapted for natural language processing to tokenize text into subword units. This is particularly useful for handling out-of-vocabulary words and improving the efficiency of tokenization in language models.

### Files in this Repository

- **`bpe.py`**:

  - Implements the `BasicTokenizer`, which performs BPE directly on UTF-8 encoded bytes.
  - This script provides a straightforward approach to tokenization by applying BPE to the byte-level representation of text.

- **`regex.py`**:
  - Utilizes a regular expression (regex) pattern to split the text based on a specified pattern before applying BPE.
  - Capable of loading `tiktoken` tokenizers, making it suitable for different GPT models.
  - This approach offers more flexibility by allowing custom text preprocessing via regex before BPE is applied.

## Installation

To use the tokenizers, simply clone this repository and import the desired tokenizer in your Python project.

```bash
git clone https://github.com/AkashKarnatak/bpe-py.git
cd bpe-py
```

## Usage

### BasicTokenizer (`bpe.py`)

The `BasicTokenizer` can be used for straightforward BPE tokenization at the byte level. This is useful when you need to tokenize text without any prior splitting or processing.

```python
from bpe import BasicTokenizer

basic_ranks = load_bpe("./basic-merges.bpe")
tokenizer = BasicTokenizer(basic_ranks)
encoded_text = tokenizer.encode("Your text here")
```

### Regex Tokenizer (`regex.py`)

The `regex.py` script allows for more advanced tokenization, where text can be preprocessed using a regex pattern before BPE is applied. Additionally, it can load `tiktoken` tokenizers compatible with various GPT models.

```python
from regex_bpe import RegexTokenizer

regex_ranks = load_bpe("./regex-merges.bpe")
tokenizer = RegexTokenizer(regex_ranks)
encoded_text = tokenizer.encode("Your text here")

# Loading a tiktoken tokenizer for a specific GPT model
gpt4_ranks = load_bpe("./o200k_base.tiktoken")
tokenizer = RegexTokenizer(regex_ranks)
encoded_text = tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉")
```

## Contributing

Contributions are welcome! If you would like to improve the BPE implementation, add new features, or fix any bugs, feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- https://www.youtube.com/watch?v=zduSFxRajkE
- https://github.com/karpathy/minbpe
- https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
