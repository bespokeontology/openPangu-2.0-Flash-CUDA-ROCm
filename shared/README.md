# shared/ - model-agnostic tooling from the openPangu gfx906 port

These files are not about openPangu. They are the parts of a gfx906 decode
engine that are the same whatever model is on the cards, extracted and verified
against two different models so another engine can take them instead of
rewriting them.

## What is here

| file | what it is |
|---|---|
| `hf_tokenizer.h` / `hf_tokenizer.cpp` | a HuggingFace `tokenizer.json` loader: byte-level BPE, vocab, merges, added tokens, and the pre-tokenizer regex read **from the JSON** |
| `tokcli.cpp` | a CLI over it: `info`, `encode`, `decode`, `check`, `ids`, `chatml`, `sample` |

```
g++ -O2 -std=c++17 hf_tokenizer.cpp tokcli.cpp -lpcre2-8 -o tokcli
tokcli <tokenizer.json> info
tokcli <tokenizer.json> encode <text-in> <tokens-out.bin>   # int64 ids, engine format
tokcli <tokenizer.json> chatml <system> <user> <tokens-out.bin>
tokcli <tokenizer.json> check  <tokens-in.bin>              # count, min, max id
tokcli <tokenizer.json> decode <tokens-in.bin>
tokcli <tokenizer.json> ids    < <decimal ids on stdin>
tokcli <tokenizer.json> sample <topk> <temp> <topp> <seed> < <logits on stdin>
```

## Verified against two models, not one

| | openPangu-2.0-Flash | Qwen3.8-27B |
|---|---|---|
| vocab / merges / added / reverse | 148899 / 148643 / 701 / 149600 | 248044 / 247587 / 33 / 248077 |
| same document -> tokens | 565 | 597 |
| id range | min 2, max 136212 | min 2, max 199956 |
| encode -> decode -> encode | idempotent | idempotent |
| round-trip on a probe sentence | ok | ok |

The sampler is deterministic for a fixed seed, and greedy (temp 0) returns the
argmax: `sample 5 0 0.9 1234` on ascending logits returns 49.

## Three defects this fixes, all of which cost real time here

1. **The pre-tokenizer regex was compiled into the source.** It is now read from
   `pre_tokenizer -> ... -> {"pattern":{"Regex":"..."}}`, including JSON unescaping
   (`\\p{L}` -> `\p{L}`, `\uXXXX` -> UTF-8).
2. **The vocabulary was sized by a constant** (`resize(151552)`, one model's padded
   head). It is now grown from the largest id the file declares, so a 248k
   vocabulary loads as happily as a 149k one.
3. **The merge table had one encoding.** Both are accepted:
   `[["left","right"], ...]` and `["left right", ...]`.

## The one habit worth copying: `check` before you trust a corpus

A prompt file tokenized by the *wrong* tokenizer loads, runs, and produces
numbers. On this project it produced ids up to 154,841 against a 151,552-row
embedding table - every id above the table indexed past the end and the run
measured timing on meaningless content. `tokcli check` prints the count and id
range; compare the max against your embedding row count before drawing any
conclusion from a long-context run.

## What is model-specific and deliberately absent

The chat template (this ships `chatml` as a worked example; Qwen has
`chat_template.jinja`), the engine's sampling defaults, and everything about a
particular architecture - attention layout, cache geometry, expert packing.
Templates and architecture do not transfer; the tokenizer, the sampler and the
id-range gate do.
