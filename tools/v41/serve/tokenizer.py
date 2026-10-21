"""DeepSeek's sequential joyai-llm pre-split, without changing Strata's tokenizer."""
import sys
from pathlib import Path

import regex

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer as BaseTokenizer


PATTERNS = (
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"""
    r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)


class Ds41Tokenizer(BaseTokenizer):
    def __init__(self, tokens, merges, token_types=None, pre="joyai-llm", **kwargs):
        if pre != "joyai-llm":
            raise ValueError("DeepSeek V4.1 requires the joyai-llm pre-tokenizer")
        super().__init__(tokens, [m for m in merges if m and not m.startswith("#version:")],
                         token_types, pre=pre, **kwargs)
        self._patterns = [regex.compile(p) for p in PATTERNS]
        self._literal = {i: t.encode("utf-8") for t, i in self.special_tokens.items()}

    def pre_split(self, text):
        pieces = [text]
        for pattern in self._patterns:
            result = []
            for piece in pieces:
                last = 0
                for match in pattern.finditer(piece):
                    if match.start() > last:
                        result.append(piece[last:match.start()])
                    if match.end() > match.start():
                        result.append(match.group())
                    last = match.end()
                if last < len(piece):
                    result.append(piece[last:])
            pieces = result
        return pieces

    def _encode_plain(self, text):
        out = []
        for piece in self.pre_split(text):
            ids = self._piece_ids.get(piece)
            if ids is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                ids = [self.ids[t] for t in self._bpe(mapped)]
                if len(self._piece_ids) < self.PIECE_CACHE_MAX:
                    self._piece_ids[piece] = ids
            out.extend(ids)
        return out

    def token_bytes(self, i):
        literal = self._literal.get(i)
        return literal if literal is not None else super().token_bytes(i)
