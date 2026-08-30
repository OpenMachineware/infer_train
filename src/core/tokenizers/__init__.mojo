# core/tokenizers/__init__.mojo
#
# Tokenizer implementations (M7).
#
# Design: one generic engine per tokenization *algorithm*; model families
# differ only in data (flavor tag, added-token table, bos/eos ids) and are
# auto-selected from the GGUF metadata by `make_tokenizer`.  Mainstream LLM
# tokenizers come in two algorithm families:
#
#   * BPE (byte-level BPE, GPT-2 style) - GPT-2/3/4, Llama, Qwen, DeepSeek,
#     Mistral, ...  -> `BpeTokenizer` (bpe_engine.mojo)
#   * SentencePiece (Unigram LM / WordPiece) - T5, Gemma, ...  -> a second
#     generic engine, to be added when a model needs it.
#
# Public surface:
#   * `make_tokenizer`      - auto-select a tokenizer from GGUF metadata
#   * `TokenizerRegistry` /
#     `register_tokenizer`  - custom-tokenizer registry
#   * `BpeTokenizer`        - the generic BPE engine
#   * `AddedToken` / `read_gguf_token_list` / `read_gguf_merge_list`

from .bpe_engine import (
    BpeTokenizer,
    AddedToken,
    read_gguf_token_list,
    read_gguf_merge_list,
)
from .registry import (
    make_tokenizer,
    TokenizerSpec,
    TokenizerRegistry,
    register_tokenizer,
)
