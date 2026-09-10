# 22 — 70,008 positions: past 2^16, and what the fast prefill actually is

## 1. The engine holds and uses 70,008 positions (measured)

`p92_chat`, `MAXPOS=262144`, 70,000 prompt positions + 8 generated, greedy, MTP off:

| quantity | value |
|---|---|
| positions ingested and decoded | **70,008** |
| steady rate over the run | 20.29 ms/token = **49.28 tok/s** |
| cost at position 65,500 | 20.82 ms |
| cost at position 70,007 | 20.87 ms |

Steps 65,500-65,507 are the first executions above 2^16 = 65,536, and they are
unremarkable: 20.81-20.85 ms, i.e. on the same flat curve as position 43,135
(20.46 ms) and 5,669 (19.83 ms). **No 16-bit position or slot index overflows
anywhere in the DSA caches, the index scores, or the selector.** That was the one
boundary that could have blocked a 262K path outright, and it is now crossed.

Note on what this run does and does not show: its token file was built with the
wrong tokenizer (receipt 21 §4), so this is a **capacity and timing** result, not a
semantic one. The semantic run is §2.

## 2. First retrieval probe on a correctly tokenized corpus

The corpus is 242,934 B of this project's own documents, tokenized with the model's
tokenizer via `tools/p92_encode.cpp` (max id 148,905, inside the 151,552-row
embedding table), wrapped in the model's chat template, with four needle sentences
inserted at 25/50/75/95% depth:

```
The amber  vault access code is 41-2739.
The basalt vault access code is 58-1904.
The cobalt vault access code is 63-8157.
The dahlia vault access code is 77-3026.
```

32,852 positions ingested at 20.21 ms/token = **49.49 tok/s**, then greedy
generation of 24 tokens. Decoded output:

```
user is asking for the access code for the basalt vault. I need to look through
the provided text to find this
```

That is the correct question being read out of a 32,852-position context and
reasoned about — the context is genuinely being attended to, not ignored — and the
answer was cut off by the 24-token budget, not by the engine. Longer-generation
runs on the same corpora are queued.

## 3. What the 698 tok/s prefill path actually is: a ceiling, not an ingest

`prefill/tests/p92_pf_bench.hip` states its own limitation, and it is decisive for
how the chunked ingest can be wired:

> DSA layers see the whole prefix. That is NOT what the model does above 2048
> positions - decode runs the indexer and attends to the top-2048 - but prefill
> launches no indexer, so this diverges from the reference engine on any prompt
> longer than 2048 and does work the model does not ask for. `PF_DSA_WIN` bounds it
> with a trailing window purely to measure the ceiling; **a window is not a
> selection and is not a correctness fix.**

So the cheap route — lift the chunked prefill into the ingest path and hand its
caches to the decoder — would write caches that are *wrong past 2,048 positions*,
because every layer's KV depends on the previous layer's output, which depends on
which positions attention read. Wiring a token file into that bench produces a fast
number and an incorrect context.

The correct chunked ingest therefore has to run the indexer and the top-2048
selection inside the chunk, at which point it is the decode semantics applied to T
rows, not the current prefill. Its prompt is also synthetic
(`tokens[i] = (START+i) % 151552`) and it persists nothing, so a real token source
and a cache handoff are needed on top of that.

## 4. Corpus tooling fix

`tools/p92_encode.cpp` (new, CPU-only, native):

```
p92_encode <tokenizer.json> encode <text-in> <tokens-out.bin>
p92_encode <tokenizer.json> chat   <text-in> <tokens-out.bin>   # chat template
p92_encode <tokenizer.json> decode <tokens-in.bin>
p92_encode <tokenizer.json> check  <tokens-in.bin>              # count, min, max id
p92_encode <tokenizer.json> ids    < <decimal ids on stdin>
```

`encode -> decode -> encode` is idempotent on real text (verified). `check` on the
old file reports max 154,841; on text tokenized by this tool, 136,212. Long-context
corpora are now built the only way that can be trusted.
