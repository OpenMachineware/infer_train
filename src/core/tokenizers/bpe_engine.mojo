# core/tokenizers/bpe_engine.mojo
#
# M7: the shared GPT-2-style byte-level BPE engine behind every built-in
# tokenizer flavor (Qwen / Llama / Hunyuan).  `BpeTokenizer` is the runtime
# type used end-to-end; the per-flavor structs in this package
# (QwenTokenizer / LlamaTokenizer / HunyuanTokenizer) construct it with the
# right flavor tag, added tokens, and bos/eos ids.
#
# Data sources:
#   * `tokenizer.json`  - vocab + merges + added tokens (streamed straight
#     out of the JSON with `JsonParser`, no DOM);
#   * the GGUF context - the complete `tokenizer.ggml.tokens` array (the
#     model's actual vocabulary, which may be *larger* than the
#     tokenizer.json coverage), plus bos/eos ids.  The GGUF array is used as
#     the decode table so every id the model can emit decodes to text.
#
# Pipeline (mirrors tokenizers-rs `Sequence[Split, ByteLevel]` + BPE):
#   1. split the text on added tokens (longest first - `<｜User｜>`,
#      `<think>`, ... match as single ids);
#   2. split the remaining segments with the GPT-2 regex
#      (`'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
#       ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+`);
#   3. byte-level encode each piece: UTF-8 bytes -> unicode chars via the
#      GPT-2 `bytes_to_unicode` map (space -> U+0120 "Ġ", ...);
#   4. greedy BPE merge (lowest-rank pair first, leftmost on ties);
#   5. look merged symbols up in the vocab.
#
# Decode reverses the byte-level map and concatenates the token strings.
#
# Flavor differences live in `TokenizerFlavor` (see core/tokenizer.mojo):
#   * qwen:    GPT-2 BPE + Qwen2/R1 added tokens (`<｜User｜>`, `<think>`, ...)
#   * llama:   plain GPT-2 BPE (llama-bpe / smaug-bpe / olmo pretokenizers)
#   * hunyuan: GPT-2 BPE + hunyuan added tokens (`<｜start▁of▁sentence｜>`, ...)
#   * custom:  user-registered vocab/merges/added-token table
#
# Notes: normalization (NFC) is not applied; the regex classes are
# approximated for bytes >= 0x80 (treated as letters).

from ..json import JsonParser, utf8_encode
from ..memory import mmap_file
from ..gguf_loader import (
    GGUFContext,
    GGUFMetaValue,
    Reader,
    get_meta_uint,
    get_meta_str,
)
from ..tokenizer import Tokenizer
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.origin import MutUntrackedOrigin
from std.collections import Span

comptime INF = 2147483647

# TokenizerFlavor tags (mirror the definitions in core/tokenizer.mojo).
comptime FLAVOR_QWEN = Int8(0)
comptime FLAVOR_LLAMA = Int8(1)
comptime FLAVOR_HUNYUAN = Int8(2)
comptime FLAVOR_GPT2 = Int8(3)
comptime FLAVOR_CUSTOM = Int8(4)


def flavor_name(flavor: Int8) -> String:
    if flavor == FLAVOR_QWEN:
        return String("qwen")
    if flavor == FLAVOR_LLAMA:
        return String("llama")
    if flavor == FLAVOR_HUNYUAN:
        return String("hunyuan")
    if flavor == FLAVOR_CUSTOM:
        return String("custom")
    return String("gpt2")


def detect_flavor(ctx: GGUFContext, bos_id: Int) -> Int8:
    """Pick the flavor from the GGUF tokenizer metadata.

    Order of precedence: `tokenizer.ggml.pre` (the llama.cpp pretokenizer
    tag) is the strongest signal, then `tokenizer.ggml.model`, then a bos-id
    heuristic (very large bos ids only occur in added-token vocabularies of
    the qwen/hunyuan families).
    """
    var pre = get_meta_str(ctx, "tokenizer.ggml.pre", String(""))
    if pre.byte_length() > 0:
        if _str_starts_with(pre, "qwen") or _str_starts_with(pre, "deepseek"):
            return FLAVOR_QWEN
        if _str_starts_with(pre, "hunyuan"):
            return FLAVOR_HUNYUAN
        if (
            _str_starts_with(pre, "llama")
            or _str_starts_with(pre, "smaug")
            or _str_starts_with(pre, "olmo")
            or _str_starts_with(pre, "vicuna")
        ):
            return FLAVOR_LLAMA
    var model = get_meta_str(ctx, "tokenizer.ggml.model", String("gpt2"))
    if model == "gpt2":
        if bos_id >= 100000:
            return FLAVOR_QWEN  # qwen/hunyuan-style added-token vocabularies
        return FLAVOR_LLAMA
    return FLAVOR_GPT2


def _str_starts_with(s: String, prefix: String) -> Bool:
    var sb = s.as_bytes()
    var pb = prefix.as_bytes()
    if len(sb) < len(pb):
        return False
    for i in range(len(pb)):
        if sb[i] != pb[i]:
            return False
    return True


def _is_special_token(token: String) -> Bool:
    """Special-token heuristic (see `derive_added_tokens`)."""
    var bytes = token.as_bytes()
    var n = len(bytes)
    if n < 3:
        return False
    if token == "<s>" or token == "</s>" or token == "<unk>":
        return True
    if token == "<pad>" or token == "<eos>" or token == "<bos>":
        return True
    if bytes[0] != UInt8(60):  # '<'
        return False
    if bytes[n - 1] != UInt8(62):  # '>'
        return False
    # "<|...|>" or "<｜...｜>" or "<[..]>" style markers
    var has_bar = False
    for i in range(1, n - 1):
        var b = bytes[i]
        if b == UInt8(124) or b == UInt8(91):  # '|' or '['
            has_bar = True
            break
        # U+FF5C ｜ = 0xEF 0xBD 0x9C (UTF-8)
        if b == UInt8(0xEF) and i + 2 < n - 1:
            if (
                bytes[i + 1] == UInt8(0xBD)
                and bytes[i + 2] == UInt8(0x9C)
            ):
                has_bar = True
                break
    return has_bar


struct AddedToken(Copyable, Movable, ImplicitlyCopyable):
    var id: Int
    var content: String
    var special: Bool

    def __init__(out self, id: Int, var content: String, special: Bool):
        self.id = id
        self.content = content^
        self.special = special


# -- byte-level encoding maps -----------------------------------------------


def _byte_to_char(byte: Int) -> Int:
    """GPT-2 bytes_to_unicode: byte -> unicode codepoint."""
    if (
        (byte >= 33 and byte <= 126)
        or (byte >= 161 and byte <= 172)
        or (byte >= 174 and byte <= 255)
    ):
        return byte
    var rank = 0
    for b in range(256):
        var printable = (
            (b >= 33 and b <= 126)
            or (b >= 161 and b <= 172)
            or (b >= 174 and b <= 255)
        )
        if b == byte:
            return 256 + rank
        if not printable:
            rank += 1
    return 256 + rank


def _char_to_byte(cp: Int) -> Int:
    """Inverse of `_byte_to_char` (unicode codepoint -> byte)."""
    if (cp >= 33 and cp <= 126) or (cp >= 161 and cp <= 172) or (
        cp >= 174 and cp <= 255
    ):
        return cp
    var n = cp - 256
    if n < 33:
        return n
    if n == 33:
        return 127
    if n < 67:
        return 128 + (n - 34)
    if n == 67:
        return 173
    return 63  # '?' fallback


# -- character classification (approximations for bytes >= 0x80) ------------


def _is_letter(byte: Int) -> Bool:
    if byte >= 65 and byte <= 90:
        return True
    if byte >= 97 and byte <= 122:
        return True
    return byte >= 128


def _is_number(byte: Int) -> Bool:
    return byte >= 48 and byte <= 57


def _is_ws(byte: Int) -> Bool:
    return (
        byte == 32
        or byte == 9
        or byte == 10
        or byte == 13
        or byte == 11
        or byte == 12
    )


struct BpeTokenizer(Movable, Tokenizer):
    var _vocab: Dict[String, Int]
    var _merges: Dict[String, Int]
    var _decode_table: List[String]
    var _added: List[AddedToken]
    var _bos_id: Int
    var _eos_id: Int
    var _sym_table: List[String]  # byte -> byte-level unicode char symbol
    var _flavor: Int8  # TokenizerFlavor tag (qwen/llama/hunyuan/gpt2/custom)
    var _add_bos: Bool  # tokenizer.ggml.add_bos_token (false by default)

    def __init__(out self):
        self._vocab = Dict[String, Int]()
        self._merges = Dict[String, Int]()
        self._decode_table = List[String]()
        self._added = List[AddedToken]()
        self._bos_id = 0
        self._eos_id = 0
        self._sym_table = List[String]()
        self._flavor = FLAVOR_GPT2
        self._add_bos = False

    # -- construction -------------------------------------------------------

    @staticmethod
    def load(
        tokenizer_json_path: String, ctx: GGUFContext
    ) raises -> Self:
        """Load tokenizer.json, using GGUF metadata for decode/bos/eos.

        The flavor is detected from the GGUF metadata (tokenizer.ggml.model
        / tokenizer.ggml.pre) unless the JSON explicitly names a model type.
        """
        var tokenizer = Self()
        tokenizer._build_sym_table()
        tokenizer._parse_tokenizer_json(tokenizer_json_path)
        tokenizer._bos_id = get_meta_uint(ctx, "tokenizer.ggml.bos_token_id", 1)
        tokenizer._eos_id = get_meta_uint(ctx, "tokenizer.ggml.eos_token_id", 2)
        tokenizer._add_bos = get_meta_uint(ctx, "tokenizer.ggml.add_bos_token", 0) != 0
        tokenizer._build_decode_table(ctx)
        tokenizer._flavor = detect_flavor(ctx, tokenizer._bos_id)
        return tokenizer^

    @staticmethod
    def load_flavor(
        tokenizer_json_path: String, ctx: GGUFContext, flavor: Int8
    ) raises -> Self:
        """`load` with an explicit flavor override."""
        var tokenizer = Self()
        tokenizer._build_sym_table()
        tokenizer._parse_tokenizer_json(tokenizer_json_path)
        tokenizer._bos_id = get_meta_uint(ctx, "tokenizer.ggml.bos_token_id", 1)
        tokenizer._eos_id = get_meta_uint(ctx, "tokenizer.ggml.eos_token_id", 2)
        tokenizer._add_bos = get_meta_uint(ctx, "tokenizer.ggml.add_bos_token", 0) != 0
        tokenizer._build_decode_table(ctx)
        tokenizer._flavor = flavor
        return tokenizer^

    @staticmethod
    def load_from_gguf(ctx: GGUFContext) raises -> Self:
        """Fallback: build the tokenizer from the GGUF vocabulary alone."""
        var tokenizer = Self()
        tokenizer._build_sym_table()
        var tokens = read_gguf_token_list(ctx)
        var merges = read_gguf_merge_list(ctx)
        tokenizer._decode_table = tokens^
        var vocab = Dict[String, Int]()
        for i in range(len(tokenizer._decode_table)):
            vocab[tokenizer._decode_table[i]] = i
        tokenizer._vocab = vocab^
        var merge_dict = Dict[String, Int]()
        for i in range(len(merges)):
            merge_dict[merges[i]] = i
        tokenizer._merges = merge_dict^
        tokenizer._bos_id = get_meta_uint(ctx, "tokenizer.ggml.bos_token_id", 1)
        tokenizer._eos_id = get_meta_uint(ctx, "tokenizer.ggml.eos_token_id", 2)
        tokenizer._add_bos = get_meta_uint(ctx, "tokenizer.ggml.add_bos_token", 0) != 0
        tokenizer._flavor = detect_flavor(ctx, tokenizer._bos_id)
        return tokenizer^

    @staticmethod
    def custom(
        vocab: Dict[String, Int],
        merges: List[String],
        added: List[AddedToken],
        bos_id: Int,
        eos_id: Int,
        flavor: Int8 = FLAVOR_CUSTOM,
    ) -> Self:
        """Build a tokenizer from caller-owned tables (the `register_tokenizer`
        path for custom tokenizers).

        `vocab` maps merged symbols (byte-level chars) to ids; `merges` is
        the ordered BPE merge list; `added` is the added-token table used
        for pre-splitting; bos/eos are the sequence boundary ids.
        """
        var tokenizer = Self()
        tokenizer._build_sym_table()
        var max_id = 0
        for token in vocab.keys():
            var id = vocab.get(token, -1)
            if id > max_id:
                max_id = id
        for a in added:
            if a.id > max_id:
                max_id = a.id
        var table = List[String]()
        for _ in range(max_id + 1):
            table.append(String(""))
        for token in vocab.keys():
            var id = vocab.get(token, -1)
            if id >= 0 and id < len(table):
                table[id] = token
        for a in added:
            if a.id >= 0 and a.id < len(table):
                table[a.id] = a.content
        tokenizer._decode_table = table^
        tokenizer._vocab = vocab
        var merge_dict = Dict[String, Int]()
        for i in range(len(merges)):
            merge_dict[merges[i]] = i
        tokenizer._merges = merge_dict^
        tokenizer._added = added
        tokenizer._bos_id = bos_id
        tokenizer._eos_id = eos_id
        tokenizer._flavor = flavor
        return tokenizer^

    def flavor(self) -> Int8:
        return self._flavor

    def flavor_name(self) -> String:
        return flavor_name(self._flavor)

    def derive_added_tokens(mut self):
        """Derive the added-token pre-split table from the decode table.

        Used when no `tokenizer.json` is available (GGUF-only models like
        Hy-MT2).  A vocab entry counts as a special token when it looks like
        a control marker: `<|...|>`, `<｜...｜>`, `[...]` pad tokens, or the
        classic BOS/EOS/UNK/PAD singles.  Same family of heuristics as
        llama.cpp's `llm_tokenizer_bpe::init`.
        """
        if len(self._added) > 0:
            return
        var added = List[AddedToken]()
        for i in range(len(self._decode_table)):
            var token = self._decode_table[i]
            if _is_special_token(token):
                added.append(AddedToken(i, token, True))
        self._added = added^

    def _build_sym_table(mut self):
        self._sym_table = List[String]()
        for b in range(256):
            var cp = _byte_to_char(b)
            var buf = unsafe_alloc[UInt8](4)
            var idx = 0
            var n = utf8_encode(cp, buf, idx)
            var span = Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=buf, length=n
            )
            self._sym_table.append(String(unsafe_from_utf8=span))

    def _parse_tokenizer_json(mut self, path: String) raises:
        var (data, size) = mmap_file(path)
        var parser = JsonParser(data, size)
        parser.skip_ws()
        parser.expect_byte(UInt8(123))  # '{'
        while True:
            parser.skip_ws()
            if parser._peek() == UInt8(125):  # '}'
                _ = parser._advance()
                break
            var key = parser.parse_string()
            parser.skip_ws()
            parser.expect_byte(UInt8(58))  # ':'
            parser.skip_ws()
            if key == "model":
                self._parse_model(parser)
            elif key == "added_tokens":
                self._parse_added_tokens(parser)
            else:
                parser.skip_value()
            parser.skip_ws()
            if parser._peek() == UInt8(44):  # ','
                _ = parser._advance()

    def _parse_model(mut self, mut parser: JsonParser) raises:
        parser.expect_byte(UInt8(123))  # '{'
        while True:
            parser.skip_ws()
            if parser._peek() == UInt8(125):  # '}'
                _ = parser._advance()
                return
            var key = parser.parse_string()
            parser.skip_ws()
            parser.expect_byte(UInt8(58))  # ':'
            parser.skip_ws()
            if key == "vocab":
                self._parse_vocab(parser)
            elif key == "merges":
                self._parse_merges(parser)
            else:
                parser.skip_value()
            parser.skip_ws()
            if parser._peek() == UInt8(44):  # ','
                _ = parser._advance()

    def _parse_vocab(mut self, mut parser: JsonParser) raises:
        parser.expect_byte(UInt8(123))  # '{'
        while True:
            parser.skip_ws()
            if parser._peek() == UInt8(125):  # '}'
                _ = parser._advance()
                return
            var token = parser.parse_string()
            parser.skip_ws()
            parser.expect_byte(UInt8(58))  # ':'
            parser.skip_ws()
            var id = parser.parse_int_raw()
            self._vocab[token] = id
            parser.skip_ws()
            if parser._peek() == UInt8(44):  # ','
                _ = parser._advance()

    def _parse_merges(mut self, mut parser: JsonParser) raises:
        parser.expect_byte(UInt8(91))  # '['
        var rank = 0
        while True:
            parser.skip_ws()
            if parser._peek() == UInt8(93):  # ']'
                _ = parser._advance()
                return
            var pair = parser.parse_string()
            self._merges[pair] = rank
            rank += 1
            parser.skip_ws()
            if parser._peek() == UInt8(44):  # ','
                _ = parser._advance()

    def _parse_added_tokens(mut self, mut parser: JsonParser) raises:
        parser.expect_byte(UInt8(91))  # '['
        while True:
            parser.skip_ws()
            if parser._peek() == UInt8(93):  # ']'
                _ = parser._advance()
                return
            parser.expect_byte(UInt8(123))  # '{'
            var id = -1
            var content = String("")
            var special = False
            while True:
                parser.skip_ws()
                if parser._peek() == UInt8(125):  # '}'
                    _ = parser._advance()
                    break
                var key = parser.parse_string()
                parser.skip_ws()
                parser.expect_byte(UInt8(58))  # ':'
                parser.skip_ws()
                if key == "id":
                    id = parser.parse_int_raw()
                elif key == "content":
                    content = parser.parse_string()
                elif key == "special":
                    special = parser.read_bool_raw()
                else:
                    parser.skip_value()
                parser.skip_ws()
                if parser._peek() == UInt8(44):  # ','
                    _ = parser._advance()
            if id >= 0 and content.byte_length() > 0:
                self._added.append(AddedToken(id, content^, special))
            parser.skip_ws()
            if parser._peek() == UInt8(44):  # ','
                _ = parser._advance()


    def _build_decode_table(mut self, ctx: GGUFContext):
        """Decode table = GGUF token list (full model vocab) + JSON
        added-token overrides; padded when the GGUF list is absent."""
        var tokens = read_gguf_token_list(ctx)
        if len(tokens) > 0:
            self._decode_table = tokens^
        else:
            # JSON-only fallback: size from the max known id.
            var max_id = 0
            for token in self._vocab.keys():
                var id = self._vocab.get(token, -1)
                if id > max_id:
                    max_id = id
            for added in self._added:
                if added.id > max_id:
                    max_id = added.id
            var table = List[String]()
            for _ in range(max_id + 1):
                table.append(String(""))
            for token in self._vocab.keys():
                var id = self._vocab.get(token, -1)
                if id >= 0 and id < len(table):
                    table[id] = token
            self._decode_table = table^
        for added in self._added:
            if added.id >= 0 and added.id < len(self._decode_table):
                self._decode_table[added.id] = added.content

    # -- encoding -----------------------------------------------------------

    def encode(self, text: String) -> List[Int]:
        return self._encode_inner(text, False)

    def encode_with_bos(self, text: String) -> List[Int]:
        return self._encode_inner(text, self._add_bos)

    def add_bos(self) -> Bool:
        return self._add_bos

    def _encode_inner(self, text: String, add_bos: Bool) -> List[Int]:
        var bytes = text.as_bytes()
        var n = len(bytes)
        var tokens = List[Int]()
        if add_bos:
            tokens.append(self._bos_id)
        var seg_start = 0
        var i = 0
        while i < n:
            var matched = -1
            var matched_len = 0
            for added in self._added:
                var alen = added.content.byte_length()
                if alen > matched_len and i + alen <= n:
                    var ok = True
                    var added_bytes = added.content.as_bytes()
                    var j = 0
                    while j < alen:
                        if added_bytes[j] != bytes[i + j]:
                            ok = False
                            break
                        j += 1
                    if ok:
                        matched = added.id
                        matched_len = alen
            if matched >= 0:
                if i > seg_start:
                    self._encode_segment(tokens, bytes, seg_start, i)
                tokens.append(matched)
                i += matched_len
                seg_start = i
            else:
                i += 1
        if n > seg_start:
            self._encode_segment(tokens, bytes, seg_start, n)
        return tokens^

    def _encode_segment(
        self,
        mut tokens: List[Int],
        bytes: Span[UInt8, _],
        start: Int,
        end: Int,
    ):
        """Regex-split [start, end) into pieces and BPE-encode each."""
        var p = start
        while p < end:
            var piece_end = self._match_piece(bytes, p, end)
            if piece_end <= p:
                p += 1  # safety: never loop forever on an unmatched byte
                continue
            self._bpe_piece(tokens, bytes, p, piece_end)
            p = piece_end

    def _match_piece(
        self, bytes: Span[UInt8, _], p: Int, end: Int
    ) -> Int:
        """Match one GPT-2 regex alternative at `p`; return the end offset.

        Alternatives are tried in regex order (leftmost wins)."""
        var b = Int(bytes[p])

        # A1: 's | 't | 're | 've | 'm | 'll | 'd  (case-insensitive)
        if b == 39:  # '\''
            if p + 1 < end:
                var c1 = Int(bytes[p + 1])
                var lower = c1 if c1 < 65 or c1 > 90 else c1 + 32
                if lower == 115:  # 's
                    return p + 2
                if lower == 116:  # 't
                    return p + 2
                if lower == 114:  # 're
                    if p + 2 < end and self._ascii_eq_nc(
                        bytes, p + 2, end, "re"
                    ):
                        return p + 3
                if lower == 118:  # 've
                    if p + 2 < end and self._ascii_eq_nc(
                        bytes, p + 2, end, "ve"
                    ):
                        return p + 3
                if lower == 109 and p + 2 <= end:  # 'm
                    return p + 2
                if lower == 108:  # 'll
                    if p + 2 < end and self._ascii_eq_nc(
                        bytes, p + 2, end, "ll"
                    ):
                        return p + 3
                if lower == 100 and p + 2 <= end:  # 'd
                    return p + 2

        # A2: [^\r\n\p{L}\p{N}]?\p{L}+
        var q = p
        if (
            not (b == 13 or b == 10)
            and not _is_letter(b)
            and not _is_number(b)
        ):
            q += 1
        var letters = 0
        while q < end and _is_letter(Int(bytes[q])):
            q += 1
            letters += 1
        if letters >= 1:
            return q

        # A3: \p{N}
        if _is_number(b):
            return p + 1

        # A4:  ?[^\s\p{L}\p{N}]+[\r\n]*
        q = p
        if b == 32:  # ' '
            q += 1
        var punct = 0
        while q < end:
            var c = Int(bytes[q])
            if _is_ws(c) or _is_letter(c) or _is_number(c):
                break
            q += 1
            punct += 1
        if punct >= 1:
            while q < end and (bytes[q] == UInt8(13) or bytes[q] == UInt8(10)):
                q += 1
            return q

        # A5: \s*[\r\n]+
        q = p
        while q < end and _is_ws(Int(bytes[q])):
            q += 1
        var r = q
        while r > p and (
            bytes[r - 1] == UInt8(13) or bytes[r - 1] == UInt8(10)
        ):
            r -= 1
        if r < q:
            return q

        # A6: \s+(?!\S) - whitespace run whose lookahead is not a non-space
        q = p
        while q < end and _is_ws(Int(bytes[q])):
            q += 1
        if q >= end:
            return q  # end of text: full run matches
        if _is_ws(Int(bytes[q])):
            return q  # followed by whitespace: full run matches
        if q - p >= 2:
            # followed by a non-space: backtrack one char so the last
            # whitespace satisfies the lookahead and stays for the next
            # piece (which may take it as its optional prefix).
            return q - 1

        # A7: \s+
        if _is_ws(b):
            return q

        return p + 1

    def _ascii_eq_nc(
        self, bytes: Span[UInt8, _], q: Int, end: Int, lit: String
    ) -> Bool:
        """Case-insensitive compare of `lit` at [q, q+len)."""
        var lb = lit.as_bytes()
        if q + len(lb) > end:
            return False
        for i in range(len(lb)):
            var c = Int(bytes[q + i])
            var lower = c if c < 65 or c > 90 else c + 32
            if lower != Int(lb[i]):
                return False
        return True

    def _bpe_piece(
        self,
        mut tokens: List[Int],
        bytes: Span[UInt8, _],
        start: Int,
        end: Int,
    ):
        var symbols = List[String]()
        for i in range(start, end):
            symbols.append(self._sym_table[Int(bytes[i])])
        if len(symbols) == 0:
            return
        # greedy BPE: lowest-rank pair, leftmost on ties
        while len(symbols) > 1:
            var best_rank = INF
            var best_i = -1
            for i in range(len(symbols) - 1):
                var pair = _pair_key(symbols[i], symbols[i + 1])
                var rank = self._merges.get(pair, INF)
                if rank < best_rank:
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            var left = symbols[best_i]
            var right = symbols[best_i + 1]
            var merged = _join_symbols(left, right)
            symbols[best_i] = merged^
            _ = symbols.pop(best_i + 1)
        for symbol in symbols:
            var id = self._vocab.get(symbol, -1)
            if id >= 0:
                tokens.append(id)

    # -- decoding -----------------------------------------------------------

    def decode(self, tokens: List[Int]) -> String:
        var total = 0
        for token in tokens:
            if token >= 0 and token < len(self._decode_table):
                total += _token_byte_len(self._decode_table[token])
        if total == 0:
            return String("")
        var buf = unsafe_alloc[UInt8](total)
        var idx = 0
        for token in tokens:
            if token >= 0 and token < len(self._decode_table):
                idx = _append_token_bytes(self._decode_table[token], buf, idx)
        var span = Span[UInt8, MutUntrackedOrigin](
            unsafe_ptr=buf, length=total
        )
        return String(unsafe_from_utf8=span)

    def vocab_size(self) -> Int:
        return len(self._decode_table)

    def bos_id(self) -> Int:
        return self._bos_id

    def eos_id(self) -> Int:
        return self._eos_id


def _pair_key(a: String, b: String) -> String:
    """BPE pair key: `a + " " + b`."""
    var with_space = a + " "
    return (with_space + b)^


def _join_symbols(a: String, b: String) -> String:
    """Merge two BPE symbols by concatenation."""
    var result = a + b
    return result^


def _append_token_bytes(
    token: String, buf: Pointer[UInt8, MutUntrackedOrigin], mut idx: Int
) -> Int:
    """Append the raw bytes a token's byte-level chars decode to.

    Chars that are part of the byte-level alphabet (printable ASCII, the
    latin-1 supplement ranges, or the mapped 256..322 range) decode through
    `_char_to_byte`; anything else (e.g. the 3-byte `｜` inside added tokens)
    passes through as raw UTF-8.
    """
    var bytes = token.as_bytes()
    var i = 0
    var n = len(bytes)
    while i < n:
        var b = Int(bytes[i])
        var cp = b
        var char_len = 1
        if b >= 0xF0 and i + 3 < n:
            cp = (
                ((b & 0x07) << 18)
                | ((Int(bytes[i + 1]) & 0x3F) << 12)
                | ((Int(bytes[i + 2]) & 0x3F) << 6)
                | (Int(bytes[i + 3]) & 0x3F)
            )
            char_len = 4
        elif b >= 0xE0 and i + 2 < n:
            cp = (
                ((b & 0x0F) << 12)
                | ((Int(bytes[i + 1]) & 0x3F) << 6)
                | (Int(bytes[i + 2]) & 0x3F)
            )
            char_len = 3
        elif b >= 0xC0 and i + 1 < n:
            cp = ((b & 0x1F) << 6) | (Int(bytes[i + 1]) & 0x3F)
            char_len = 2
        var is_byte_level = (
            (cp >= 33 and cp <= 126)
            or (cp >= 161 and cp <= 172)
            or (cp >= 174 and cp <= 255)
            or (cp >= 256 and cp <= 322)
        )
        if is_byte_level:
            buf.unsafe_store(idx, UInt8(_char_to_byte(cp)))
            idx += 1
        else:
            for j in range(char_len):
                buf.unsafe_store(idx, bytes[i + j])
                idx += 1
        i += char_len
    return idx


def _token_byte_len(token: String) -> Int:
    """Number of raw bytes `_append_token_bytes` would emit."""
    var bytes = token.as_bytes()
    var i = 0
    var n = len(bytes)
    var total = 0
    while i < n:
        var b = Int(bytes[i])
        var cp = b
        var char_len = 1
        if b >= 0xF0 and i + 3 < n:
            cp = (
                ((b & 0x07) << 18)
                | ((Int(bytes[i + 1]) & 0x3F) << 12)
                | ((Int(bytes[i + 2]) & 0x3F) << 6)
                | (Int(bytes[i + 3]) & 0x3F)
            )
            char_len = 4
        elif b >= 0xE0 and i + 2 < n:
            cp = (
                ((b & 0x0F) << 12)
                | ((Int(bytes[i + 1]) & 0x3F) << 6)
                | (Int(bytes[i + 2]) & 0x3F)
            )
            char_len = 3
        elif b >= 0xC0 and i + 1 < n:
            cp = ((b & 0x1F) << 6) | (Int(bytes[i + 1]) & 0x3F)
            char_len = 2
        var is_byte_level = (
            (cp >= 33 and cp <= 126)
            or (cp >= 161 and cp <= 172)
            or (cp >= 174 and cp <= 255)
            or (cp >= 256 and cp <= 322)
        )
        if is_byte_level:
            total += 1
        else:
            total += char_len
        i += char_len
    return total


# -- GGUF vocabulary helpers ------------------------------------------------


def read_gguf_token_list(context: GGUFContext) -> List[String]:
    """Read the `tokenizer.ggml.tokens` string array (ordered by id)."""
    var value = context.metadata.get(
        "tokenizer.ggml.tokens", GGUFMetaValue()
    )
    if value.kind != 5 or value.arr_len <= 0:
        return List[String]()
    var reader = Reader(context.data)
    reader.offset = value.arr_offset
    var tokens = List[String]()
    for i in range(value.arr_len):
        tokens.append(reader.read_string())
    return tokens^


def read_gguf_merge_list(context: GGUFContext) -> List[String]:
    """Read the `tokenizer.ggml.merges` string array (ordered by rank)."""
    var value = context.metadata.get(
        "tokenizer.ggml.merges", GGUFMetaValue()
    )
    if value.kind != 5 or value.arr_len <= 0:
        return List[String]()
    var reader = Reader(context.data)
    reader.offset = value.arr_offset
    var merges = List[String]()
    for i in range(value.arr_len):
        merges.append(reader.read_string())
    return merges^
